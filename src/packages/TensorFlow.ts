import * as tf from '@tensorflow/tfjs'
import { convertCanvasDataToTensor } from './utils/canvas'
import { Metadata } from '../types/TensorFlow'
import { CLASSES } from './label'

const isMetadata = (c: any): c is Metadata => !!c && Array.isArray(c.labels)

const fillMetadata = (data: Partial<Metadata>) => {
  data.packageVersion = data.packageVersion || 'untitled package version'
  data.packageName = data.packageName || 'untitled package name'
  data.timeStamp = data.timeStamp || new Date().toISOString()
  data.userMetadata = data.userMetadata || {}
  data.modelName = data.modelName || 'untitled model name'
  data.labels = data.labels || []
  data.imageSize = data.imageSize || 224

  return data as Metadata
}

const analyzeMetadata = async (metadata: string | Metadata) => {
  let metadataJSON: Metadata

  if (typeof metadata === 'string') {
    const metadataResponse = await fetch(metadata + 'metadata.json')
    metadataJSON = await metadataResponse.json()
  } else if (isMetadata(metadata + 'metadata.json')) {
    metadataJSON = metadata
  } else {
    throw new Error('Invalid Metadata provided')
  }

  return fillMetadata(metadataJSON)
}

interface DetectionStyle {
  fontSize: string
  fontFamily: string
  labelStrokeColor: string
  labelBackgroundColor: string
}

const defaultDetectionStyle = {
  fontSize: '12px',
  fontFamily: 'sans-serif',
  labelStrokeColor: '#0F0',
  labelBackgroundColor: '#0F0',
}

export const predictObjects = async (targetId: string, detectionStyle: DetectionStyle = defaultDetectionStyle) => {
  await tf.ready()

  const modelURL = 'https://tfhub.dev/tensorflow/tfjs-model/ssd_mobilenet_v2/1/default/1'
  const model = await loadGraphModelFromURL(modelURL)

  const targetElement = document.getElementById(targetId) as HTMLVideoElement | HTMLCanvasElement | HTMLImageElement
  const tensorImage = tf.browser.fromPixels(targetElement)

  // [1, height, width, 3]の形状に変更
  const readyfied = tf.expandDims(tensorImage, 0)
  const results: any = await model.executeAsync(readyfied)

  const detectionCanvas = document.getElementById('detection') as HTMLCanvasElement
  const ctx = detectionCanvas.getContext('2d')
  const imgWidth = targetElement.width
  const imgHeight = targetElement.height

  detectionCanvas.width = imgWidth
  detectionCanvas.height = imgHeight

  const { fontSize, fontFamily, labelStrokeColor, labelBackgroundColor } = detectionStyle

  if (ctx) {
    ctx.font = fontSize + ' ' + fontFamily
    ctx.textBaseline = 'top'
  }

  const detectionThreshold = 0.3
  const iouThreshold = 0.5
  const maxBoxes = 20

  // results[0]: 最大1の確率値を90個持つ1,917個の検出結果
  const prominentDetection = tf.topk(results[0])

  const justBoxes = results[1].squeeze()
  const justValues = prominentDetection.values.squeeze()

  const [maxIndices, scores, boxes] = await Promise.all([prominentDetection.indices.data(), justValues.array(), justBoxes.array()])

  const nmsDetections = await tf.image.nonMaxSuppressionWithScoreAsync(justBoxes, justValues, maxBoxes, iouThreshold, detectionThreshold, 1)

  const chosen = await nmsDetections.selectedIndices.data()

  tf.dispose([
    results[0],
    results[1],
    model,
    nmsDetections.selectedIndices,
    nmsDetections.selectedScores,
    prominentDetection.indices,
    prominentDetection.values,
    tensorImage,
    readyfied,
    justBoxes,
    justValues,
  ])

  const classes = [] as any

  chosen.forEach((detection) => {
    if (ctx) {
      ctx.strokeStyle = labelStrokeColor
      ctx.lineWidth = 4
      ctx.globalCompositeOperation = 'destination-over'

      const detectedIndex = maxIndices[detection]
      const detectedClass = CLASSES[detectedIndex]
      const detectedScore = scores[detection]
      const dBox = boxes[detection]

      classes.push({
        className: detectedClass,
        probability: detectedScore,
      })

      const startY = dBox[0] > 0 ? dBox[0] * imgHeight : 0
      const startX = dBox[1] > 0 ? dBox[1] * imgWidth : 0
      const height = (dBox[2] - dBox[0]) * imgHeight
      const width = (dBox[3] - dBox[1]) * imgWidth
      ctx.strokeRect(startX, startY, width, height)

      ctx.globalCompositeOperation = 'source-over'
      ctx.fillStyle = labelBackgroundColor

      const textHeight = 16
      const textPad = 4
      const label = `${detectedClass} ${Math.round(detectedScore * 100)}%`
      const textWidth = ctx.measureText(label).width

      ctx.fillRect(startX, startY, textWidth + textPad, textHeight + textPad)
      ctx.fillStyle = '#000000'
      ctx.fillText(label, startX, startY)
    }
  })

  return classes
}

export class TensorFlow {
  protected parsedMetaData: Metadata

  constructor(public model: tf.LayersModel, metaData: Partial<Metadata>) {
    this.parsedMetaData = fillMetadata(metaData)
  }

  getMetaDatalabel() {
    return this.parsedMetaData.labels
  }

  async predict(image: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement) {
    const logits = tf.tidy(() => {
      const croppedImage = convertCanvasDataToTensor(image)
      return this.model.predict(croppedImage)
    })

    const values = await (logits as tf.Tensor<tf.Rank>).data()

    const classes = []

    for (let i = 0; i < values.length; i++) {
      classes.push({
        className: this.parsedMetaData.labels[i],
        probability: values[i],
      })
    }

    tf.dispose(logits)

    return classes
  }
}

export async function loadGraphModelFromURL(modelURL: string) {
  const customModel = await tf.loadGraphModel(modelURL, { fromTFHub: true })
  return customModel
}

export async function loadFromURL(modelURL: string, metadata?: string | Metadata) {
  const customModel = await tf.loadLayersModel(modelURL + 'model.json')
  const metadataJSON = metadata ? await analyzeMetadata(metadata) : {}
  return new TensorFlow(customModel, metadataJSON)
}

export async function loadFromFiles(model: File, weights: File, metadata: File) {
  const customModel = await tf.loadLayersModel(tf.io.browserFiles([model, weights]))
  const metadataFile = await new Response(metadata).json()
  const metadataJSON = metadata ? await analyzeMetadata(metadataFile) : {}
  return new TensorFlow(customModel, metadataJSON)
}

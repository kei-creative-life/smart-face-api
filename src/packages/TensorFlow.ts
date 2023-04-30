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

export const predictSSD = async () => {
  await tf.ready()
  const modelPath = 'https://tfhub.dev/tensorflow/tfjs-model/ssd_mobilenet_v2/1/default/1'

  const model = await tf.loadGraphModel(modelPath, { fromTFHub: true })
  const mysteryImage = document.getElementById('mask') as HTMLImageElement
  const myTensor = tf.browser.fromPixels(mysteryImage)

  // [1, height, width, 3]の形状に変更
  const readyfied = tf.expandDims(myTensor, 0)

  const results: any = await model.executeAsync(readyfied)

  // Prep Canvas
  const detection = document.getElementById('detection') as HTMLCanvasElement
  const ctx = detection.getContext('2d')
  const imgWidth = mysteryImage.width
  const imgHeight = mysteryImage.height
  detection.width = imgWidth
  detection.height = imgHeight
  if (ctx) {
    ctx.font = '16px sans-serif'
    ctx.textBaseline = 'top'
  }

  // Get a clean tensor of indices
  const detectionThreshold = 0.3
  const iouThreshold = 0.5
  const maxBoxes = 20
  // results[0]: 最大1の確率値を90個持つ1,917個の検出結果
  const prominentDetection = tf.topk(results[0])
  prominentDetection.indices.print()
  prominentDetection.values.print()
  const justBoxes = results[1].squeeze()
  const justValues = prominentDetection.values.squeeze()

  // Move results back to JavaScript in parallel
  const [maxIndices, scores, boxes] = await Promise.all([prominentDetection.indices.data(), justValues.array(), justBoxes.array()])

  // https://arxiv.org/pdf/1704.04503.pdf, use Async to keep visuals
  const nmsDetections = await tf.image.nonMaxSuppressionWithScoreAsync(
    justBoxes, // [numBoxes, 4]
    justValues, // [numBoxes]
    maxBoxes,
    iouThreshold,
    detectionThreshold,
    1 // 0 is normal NMS, 1 is Soft-NMS for overlapping support
  )

  const chosen = await nmsDetections.selectedIndices.data()
  // Mega Clean
  tf.dispose([
    results[0],
    results[1],
    model,
    nmsDetections.selectedIndices,
    nmsDetections.selectedScores,
    prominentDetection.indices,
    prominentDetection.values,
    myTensor,
    readyfied,
    justBoxes,
    justValues,
  ])

  chosen.forEach((detection) => {
    if (ctx) {
      ctx.strokeStyle = '#0F0'
      ctx.lineWidth = 4
      ctx.globalCompositeOperation = 'destination-over'
      const detectedIndex = maxIndices[detection]
      const detectedClass = CLASSES[detectedIndex]
      const detectedScore = scores[detection]
      const dBox = boxes[detection]

      // No negative values for start positions
      const startY = dBox[0] > 0 ? dBox[0] * imgHeight : 0
      const startX = dBox[1] > 0 ? dBox[1] * imgWidth : 0
      const height = (dBox[2] - dBox[0]) * imgHeight
      const width = (dBox[3] - dBox[1]) * imgWidth
      ctx.strokeRect(startX, startY, width, height)
      // Draw the label background.
      ctx.globalCompositeOperation = 'source-over'
      ctx.fillStyle = '#0B0'
      const textHeight = 16
      const textPad = 4
      const label = `${detectedClass} ${Math.round(detectedScore * 100)}%`
      const textWidth = ctx.measureText(label).width
      ctx.fillRect(startX, startY, textWidth + textPad, textHeight + textPad)
      // Draw the text last to ensure it's on top.
      ctx.fillStyle = '#000000'
      ctx.fillText(label, startX, startY)
    }
  })
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
    // await this.predictSSD()

    const logits = tf.tidy(() => {
      const croppedImage = convertCanvasDataToTensor(image)
      return this.model.predict(croppedImage)
    })

    const values = await (logits as tf.Tensor<tf.Rank>).data()

    const classes = []

    console.log(values)

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

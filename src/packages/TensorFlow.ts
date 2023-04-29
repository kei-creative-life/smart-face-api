import * as tf from '@tensorflow/tfjs'

export class TensorFlow {
  constructor(public model: tf.LayersModel) {}

  async predict(image: any) {
    const logits = tf.tidy(() => {
      return this.model.predict(image)
    })

    const values = await (logits as tf.Tensor<tf.Rank>).data()

    const classes = []

    for (let i = 0; i < values.length; i++) {
      classes.push({
        // className: this.metadata.labels[i],
        probability: values[i],
      })
    }

    console.log(classes)

    tf.dispose(logits)

    return classes
  }
}

export function cropTensor(img: tf.Tensor3D): tf.Tensor3D {
  const imageSize = 224
  // モデルで指定されたサイズにリサイズする
  const imageResize = tf.image.resizeBilinear(img, [imageSize, imageSize], false)
  const size = Math.min(imageResize.shape[0], img.shape[1])
  const centerHeight = imageResize.shape[0] / 2
  const beginHeight = centerHeight - size / 2
  const centerWidth = imageResize.shape[1] / 2
  const beginWidth = centerWidth - size / 2
  return imageResize.slice([beginHeight, beginWidth, 0], [size, size, 3])
}

export async function convertCanvasDataToTensor(rasterElement: HTMLVideoElement | HTMLCanvasElement): Promise<any> {
  // tf.tidy()によってメモリリークを防ぐ
  return tf.tidy(() => {
    // canvas要素に表示された画像データを読み込んで、tf.TensorというTensor形式に変換して返却する
    const pixels = tf.browser.fromPixels(rasterElement)
    // crop the image so we're using the center square
    const cropped = cropTensor(pixels)
    // Tensorのshapeを[224, 224, 3]から[1, 224, 224, 3]に変換する
    const batchedImage = cropped.expandDims(0)
    // Normalize the image between -1 and a1. The image comes in between 0-255
    // so we divide by 127 and subtract 1.
    // int32からfloat32に変換する
    return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1))
  })
}

export async function loadFromURL(model: string) {
  const customModel = await tf.loadLayersModel(model + 'model.json')
  return new TensorFlow(customModel)
}

export async function loadFromFiles(model: File, weights: File) {
  const customModel = await tf.loadLayersModel(tf.io.browserFiles([model, weights]))
  return new TensorFlow(customModel)
}

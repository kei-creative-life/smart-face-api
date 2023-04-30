import * as tf from '@tensorflow/tfjs'

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

export function convertCanvasDataToTensor(rasterElement: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement) {
  return tf.tidy(() => {
    // canvas要素に表示された画像データを読み込んで、tf.TensorというTensor形式に変換して返却する
    const pixels = tf.browser.fromPixels(rasterElement)
    // crop the image so we're using the center square
    const cropped = cropTensor(pixels)
    // Tensorのshapeを[224, 224, 3]から[1, 224, 224, 3]に変換する
    const batchedImage = cropped.expandDims(0)
    // Normalize the image between -1 and a1. The image comes in between 0-255
    // so we divide by 127 and subtract 1.
    return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1))
  })
}

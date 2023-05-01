import * as TensorFlowModule from './TensorFlow'

export { TensorFlowModule }

// SSD Test
const getImage = document.getElementById('sample') as HTMLImageElement
// const defaultDetectionStyle = {
//   fontSize: '16px',
//   fontFamily: 'sans-serif',
//   labelStrokeColor: 'red',
//   labelBackgroundColor: 'red',
// }
// const test = await TensorFlowModule.predictObjects('sample')
// console.log(test)

// Face Detect Test
// const faces = await TensorFlowModule.runFaceDetect(getImage)
// const box = faces[0].box
// const { height, width, xMax, xMin, yMax, yMin } = box

const canvas = document.getElementById('detection') as HTMLCanvasElement
if (canvas && canvas.getContext) {
  const ctx = canvas.getContext('2d')

  if (ctx) {
    // ctx.fillRect(25, 25, 100, 100)
    // ctx.clearRect(45, 45, 60, 60)
    ctx.strokeRect(100, 50, 100, 115)
  }
}
// 0
// :
// box
// :
// height
// :
// 115.18585057216961
// width
// :
// 100.89765388276578
// xMax
// :
// 380.86267064689235
// xMin
// :
// 279.9650167641266
// yMax
// :
// 202.05385323484228
// yMin
// :
// 86.86800266267267

// Detect Model Test
// const URL = 'https://teachablemachine.withgoogle.com/models/WdEY9SIhG/'
// const model = await TensorFlowModule.loadFromURL(URL, URL)
// const crop = await SmartFaceAPIModule.convertCanvasDataToTensor(getImage)
// const result = await model

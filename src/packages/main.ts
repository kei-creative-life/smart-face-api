import * as TensorFlowModule from './TensorFlow'

export { TensorFlowModule }

const getImage = document.getElementById('sample') as HTMLImageElement
const canvas = document.getElementById('detection') as HTMLCanvasElement

// Bokeh Image
TensorFlowModule.blurImage(getImage, canvas)

// SSD Test
// const defaultDetectionStyle = {
//   fontSize: '16px',
//   fontFamily: 'sans-serif',
//   labelStrokeColor: 'red',
//   labelBackgroundColor: 'red',
// }
// const test = await TensorFlowModule.predictObjects('sample')
// console.log(test)

// Detect Model Test
// const URL = 'https://teachablemachine.withgoogle.com/models/WdEY9SIhG/'
// const model = await TensorFlowModule.loadFromURL(URL, URL)
// const crop = await SmartFaceAPIModule.convertCanvasDataToTensor(getImage)
// const result = await model

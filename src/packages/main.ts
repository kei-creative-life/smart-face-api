import * as TensorFlowModule from './TensorFlow'

export { TensorFlowModule }

// SSD Test
// const getImage = document.getElementById('mask') as HTMLImageElement
const defaultDetectionStyle = {
  fontSize: '16px',
  fontFamily: 'sans-serif',
  labelStrokeColor: 'red',
  labelBackgroundColor: 'red',
}
const test = await TensorFlowModule.predictObjects('sample', defaultDetectionStyle)
console.log(test)

// Detect Model Test
// const URL = 'https://teachablemachine.withgoogle.com/models/WdEY9SIhG/'
// const model = await TensorFlowModule.loadFromURL(URL, URL)
// const crop = await SmartFaceAPIModule.convertCanvasDataToTensor(getImage)
// const result = await model

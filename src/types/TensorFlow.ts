export interface Metadata {
  tfjsVersion: string
  tmVersion?: string
  packageVersion: string
  packageName: string
  modelName?: string
  timeStamp?: string
  labels: string[]
  userMetadata?: {}
  grayscale?: boolean
  imageSize?: number
}

import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
})

export interface ScenarioInput {
  price: number
  marketing_spend: number
  num_features: number
  usage: number
  impressions: number
  clicks: number
  text: string
}

export interface SimulationResult {
  pricing?: {
    demand: number
    revenue: number
    price_change_pct: number
    elasticity: number
    shap_drivers: Array<{ feature: string; importance: number; direction: string }>
  }
  marketing?: {
    conversion_rate: number
    engagement: number
    shap_drivers: Array<{ feature: string; importance: number; direction: string }>
  }
  sentiment?: {
    sentiment_score: number
    label: string
    context_adjustments: object
  }
  churn?: {
    churn_probability: number
    risk_level: string
    shap_drivers: Array<{ feature: string; importance: number; direction: string }>
  }
  propagation_trace?: Record<string, number>
}

export const simulate = (scenario: ScenarioInput) =>
  api.post<SimulationResult>('/simulate', scenario)

export const simulateCompare = (baseline: ScenarioInput, scenario: ScenarioInput) =>
  api.post('/simulate/compare', { baseline, scenario, text: scenario.text })

export const chat = (message: string) =>
  api.post('/chat', { message })

export const getStatus = () => api.get('/status')

export const uploadDataset = (file: File) => {
  const formData = new FormData()
  formData.append('file', file)
  return api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

export const listDatasets = () => api.get('/upload/datasets')

export const trainModel = (datasetPath: string, modelName: string, targetColumn?: string) =>
  api.post('/train', { dataset_path: datasetPath, model_name: modelName, target_column: targetColumn })

export const trainAll = () => api.post('/train/all')

export const getCounterfactuals = (modelName: string, scenario: object, totalCfs: number = 4) =>
  api.post('/counterfactual', { model_name: modelName, scenario, total_cfs: totalCfs })

export default api

import axios from 'axios'
import { supabase } from './supabase'

const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
})

// Auth interceptor — attach JWT to every request
api.interceptors.request.use(async (config) => {
  const { data: { session } } = await supabase.auth.getSession()
  if (session?.access_token) {
    config.headers.Authorization = `Bearer ${session.access_token}`
  }
  return config
})

// Response interceptor — handle 401s
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      const { error: refreshError } = await supabase.auth.refreshSession()
      if (refreshError) {
        await supabase.auth.signOut()
        window.location.href = '/login'
      }
    }
    return Promise.reject(error)
  }
)

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

// Existing endpoints
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

// Session endpoints
export const sessionsApi = {
  list: () => api.get('/sessions'),
  create: (title?: string) => api.post('/sessions', { title }),
  get: (id: string) => api.get(`/sessions/${id}`),
  update: (id: string, data: { title?: string; has_uploaded_data?: boolean }) =>
    api.patch(`/sessions/${id}`, data),
  delete: (id: string) => api.delete(`/sessions/${id}`),
  messages: (id: string) => api.get(`/sessions/${id}/messages`),
}

// Knowledge base endpoints
export const knowledgeApi = {
  list: () => api.get('/knowledge'),
  upload: (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post('/knowledge/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
  delete: (docId: string) => api.delete(`/knowledge/${docId}`),
}

// Streaming chat — returns a ReadableStream
export async function streamChat(message: string, sessionId: string): Promise<ReadableStream<string>> {
  const { data: { session } } = await supabase.auth.getSession()
  const token = session?.access_token || ''

  const response = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
    body: JSON.stringify({ message, session_id: sessionId }),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Stream failed' }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }

  return new ReadableStream({
    async start(controller) {
      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            try {
              const parsed = JSON.parse(data)
              if (parsed.token) {
                controller.enqueue(parsed.token)
              } else if (parsed.done) {
                controller.close()
                return
              } else if (parsed.error) {
                controller.enqueue(`\n\n**Error:** ${parsed.error}`)
                controller.close()
                return
              }
            } catch {}
          }
        }
      }
      controller.close()
    }
  })
}

export default api

import { useQuery } from '@tanstack/react-query'
import { getStatus, trainAll } from '../api/client'
import { Activity, CheckCircle, XCircle, Zap } from 'lucide-react'
import { useState } from 'react'
import toast from 'react-hot-toast'

export default function Dashboard() {
  const { data, isLoading, refetch } = useQuery({
    queryKey: ['status'],
    queryFn: () => getStatus().then(r => r.data),
  })
  const [training, setTraining] = useState(false)

  const handleTrainAll = async () => {
    setTraining(true)
    try {
      await trainAll()
      toast.success('All models trained successfully!')
      refetch()
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Training failed')
    } finally {
      setTraining(false)
    }
  }

  const models = data?.models || {}

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Dashboard</h2>
          <p className="text-gray-600 mt-1">AI Business Decision Simulation System</p>
        </div>
        <button
          onClick={handleTrainAll}
          disabled={training}
          className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 transition-colors"
        >
          <Zap size={18} />
          {training ? 'Training...' : 'Train All Models'}
        </button>
      </div>

      {/* Model Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {Object.entries(models).map(([name, info]: [string, any]) => (
          <ModelCard key={name} name={name} info={info} />
        ))}
        {isLoading && [1, 2, 3, 4].map(i => (
          <div key={i} className="bg-white rounded-xl p-6 border border-gray-200 animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-1/2 mb-4" />
            <div className="h-8 bg-gray-200 rounded w-3/4" />
          </div>
        ))}
      </div>

      {/* Architecture Overview */}
      <div className="bg-white rounded-xl p-6 border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Causal Propagation Flow</h3>
        <div className="flex items-center justify-center gap-4 flex-wrap">
          <FlowNode label="Pricing" color="blue" />
          <Arrow />
          <FlowNode label="Marketing" color="green" />
          <Arrow />
          <FlowNode label="Sentiment" color="yellow" />
          <Arrow />
          <FlowNode label="Churn" color="red" />
        </div>
        <p className="text-sm text-gray-500 text-center mt-4">
          Models execute sequentially. Each model's output feeds as input to downstream models.
        </p>
      </div>

      {/* Quick Start */}
      <div className="bg-gradient-to-r from-primary-50 to-blue-50 rounded-xl p-6 border border-primary-100">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Quick Start</h3>
        <ol className="list-decimal list-inside space-y-2 text-gray-700">
          <li>Click <strong>"Train All Models"</strong> to train on sample data</li>
          <li>Go to <strong>Simulate</strong> to run what-if scenarios with sliders</li>
          <li>Check <strong>What-If</strong> for counterfactual recommendations</li>
          <li>Use <strong>Chat</strong> to ask natural language questions</li>
          <li>Upload your own data in <strong>Data</strong> tab</li>
        </ol>
      </div>
    </div>
  )
}

function ModelCard({ name, info }: { name: string; info: any }) {
  const trained = info?.trained
  const metrics = info?.metrics || {}
  const primaryMetric = metrics.accuracy || metrics.r2_score || metrics.accuracy_on_sample

  return (
    <div className="bg-white rounded-xl p-6 border border-gray-200 hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-gray-600 capitalize">{name}</span>
        {trained ? (
          <CheckCircle className="w-5 h-5 text-green-500" />
        ) : (
          <XCircle className="w-5 h-5 text-gray-300" />
        )}
      </div>
      <div className="flex items-end gap-2">
        <span className="text-2xl font-bold text-gray-900">
          {primaryMetric ? `${(primaryMetric * 100).toFixed(1)}%` : '--'}
        </span>
        {trained && (
          <span className="text-xs text-green-600 font-medium mb-1">Trained</span>
        )}
      </div>
      {metrics.n_samples && (
        <p className="text-xs text-gray-400 mt-2">{metrics.n_samples} samples</p>
      )}
    </div>
  )
}

function FlowNode({ label, color }: { label: string; color: string }) {
  const colors: Record<string, string> = {
    blue: 'bg-blue-100 text-blue-800 border-blue-200',
    green: 'bg-green-100 text-green-800 border-green-200',
    yellow: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    red: 'bg-red-100 text-red-800 border-red-200',
  }
  return (
    <div className={`px-4 py-2 rounded-lg border font-medium text-sm ${colors[color]}`}>
      {label}
    </div>
  )
}

function Arrow() {
  return <span className="text-gray-400 text-xl">→</span>
}

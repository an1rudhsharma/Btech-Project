import { useState } from 'react'
import { getCounterfactuals } from '../api/client'
import { GitBranch, ArrowRight, Lightbulb } from 'lucide-react'
import toast from 'react-hot-toast'

export default function Counterfactuals() {
  const [scenario, setScenario] = useState({
    price: 150,
    marketing_spend: 3000,
    num_features: 3,
    usage: 30,
    tenure: 6,
    satisfaction: 2.5,
  })
  const [results, setResults] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [modelName, setModelName] = useState('churn')

  const handleGenerate = async () => {
    setLoading(true)
    try {
      const res = await getCounterfactuals(modelName, scenario, 4)
      setResults(res.data)
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Failed to generate counterfactuals. Train the model first.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Counterfactual Explorer</h2>
        <p className="text-gray-600 mt-1">
          Find the minimum changes needed to flip a prediction. "What should I change to prevent churn?"
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Input */}
        <div className="bg-white rounded-xl p-6 border border-gray-200 space-y-4">
          <h3 className="font-semibold text-gray-900">Current Scenario (Bad Outcome)</h3>
          <div>
            <label className="text-sm text-gray-600">Model</label>
            <select
              value={modelName}
              onChange={e => setModelName(e.target.value)}
              className="w-full mt-1 border border-gray-300 rounded-lg px-3 py-2 text-sm"
            >
              <option value="churn">Churn (Classification)</option>
              <option value="marketing">Marketing (Regression)</option>
              <option value="pricing">Pricing (Regression)</option>
            </select>
          </div>
          {Object.entries(scenario).map(([key, val]) => (
            <div key={key}>
              <div className="flex justify-between">
                <label className="text-sm text-gray-600">{key.replace('_', ' ')}</label>
                <span className="text-sm font-medium">{val}</span>
              </div>
              <input
                type="range"
                min={key === 'satisfaction' ? 1 : key === 'tenure' ? 1 : 0}
                max={key === 'price' ? 300 : key === 'marketing_spend' ? 20000 : key === 'satisfaction' ? 5 : key === 'tenure' ? 72 : 100}
                step={key === 'satisfaction' ? 0.5 : key === 'marketing_spend' ? 500 : 1}
                value={val}
                onChange={e => setScenario(prev => ({ ...prev, [key]: Number(e.target.value) }))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-red-500"
              />
            </div>
          ))}
          <button
            onClick={handleGenerate}
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50"
          >
            <GitBranch size={16} />
            {loading ? 'Generating...' : 'Find Alternatives'}
          </button>
        </div>

        {/* Results */}
        <div className="lg:col-span-2">
          {results?.status === 'success' ? (
            <div className="space-y-4">
              <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4 flex items-start gap-3">
                <Lightbulb className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-yellow-800">
                    Found {results.total_generated} alternative scenarios
                  </p>
                  <p className="text-sm text-yellow-700 mt-1">
                    These are the minimum changes needed to flip the prediction outcome.
                  </p>
                </div>
              </div>

              {results.counterfactuals.map((cf: any, idx: number) => (
                <div key={idx} className="bg-white rounded-xl p-5 border border-gray-200 hover:shadow-md transition-shadow">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-semibold text-gray-900">Alternative {idx + 1}</span>
                    <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full font-medium">
                      Feasibility: {(cf.feasibility * 100).toFixed(0)}%
                    </span>
                  </div>

                  <div className="space-y-2">
                    {cf.changes.map((change: any, cIdx: number) => (
                      <div key={cIdx} className="flex items-center gap-3 text-sm">
                        <span className="text-gray-600 w-32">{change.feature.replace('_', ' ')}</span>
                        <span className="font-medium text-red-600">{change.from}</span>
                        <ArrowRight size={14} className="text-gray-400" />
                        <span className="font-medium text-green-600">{change.to}</span>
                        <span className="text-xs text-gray-400">({change.change_pct > 0 ? '+' : ''}{change.change_pct}%)</span>
                      </div>
                    ))}
                  </div>

                  {cf.new_predicted_outcome !== null && (
                    <div className="mt-3 pt-3 border-t border-gray-100">
                      <span className="text-xs text-gray-500">New predicted outcome: </span>
                      <span className="text-sm font-semibold text-green-700">{cf.new_predicted_outcome}</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : results?.status === 'not_available' ? (
            <EmptyState message={results.message} />
          ) : results?.status === 'error' ? (
            <EmptyState message={`Error: ${results.message}`} />
          ) : (
            <EmptyState message="Set a scenario with a bad outcome (e.g., high price, low features) and click 'Find Alternatives' to see what to change." />
          )}
        </div>
      </div>
    </div>
  )
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="bg-white rounded-xl p-12 border border-gray-200 text-center">
      <GitBranch className="w-12 h-12 text-gray-300 mx-auto mb-4" />
      <p className="text-gray-500">{message}</p>
    </div>
  )
}

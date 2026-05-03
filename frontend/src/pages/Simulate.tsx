import { useState } from 'react'
import { simulate, simulateCompare, ScenarioInput, SimulationResult } from '../api/client'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, Radar, Legend } from 'recharts'
import { Play, RotateCcw } from 'lucide-react'
import toast from 'react-hot-toast'

const DEFAULT_SCENARIO: ScenarioInput = {
  price: 100,
  marketing_spend: 5000,
  num_features: 5,
  usage: 50,
  impressions: 10000,
  clicks: 500,
  text: 'Product is good',
}

export default function Simulate() {
  const [scenario, setScenario] = useState<ScenarioInput>({ ...DEFAULT_SCENARIO })
  const [results, setResults] = useState<SimulationResult | null>(null)
  const [comparison, setComparison] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const handleSimulate = async () => {
    setLoading(true)
    try {
      const res = await simulate(scenario)
      setResults(res.data)
      // Also get comparison with baseline
      const comp = await simulateCompare(DEFAULT_SCENARIO, scenario)
      setComparison(comp.data)
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Simulation failed. Train models first.')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setScenario({ ...DEFAULT_SCENARIO })
    setResults(null)
    setComparison(null)
  }

  const updateParam = (key: keyof ScenarioInput, value: number | string) => {
    setScenario(prev => ({ ...prev, [key]: value }))
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Scenario Simulator</h2>
          <p className="text-gray-600 mt-1">Adjust parameters and see predictions cascade through the causal graph</p>
        </div>
        <div className="flex gap-2">
          <button onClick={handleReset} className="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50">
            <RotateCcw size={16} /> Reset
          </button>
          <button onClick={handleSimulate} disabled={loading} className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50">
            <Play size={16} /> {loading ? 'Running...' : 'Simulate'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <div className="bg-white rounded-xl p-6 border border-gray-200 space-y-5">
          <h3 className="font-semibold text-gray-900">Scenario Parameters</h3>
          <SliderControl label="Price" value={scenario.price} min={10} max={500} step={5} onChange={v => updateParam('price', v)} unit="$" />
          <SliderControl label="Marketing Spend" value={scenario.marketing_spend} min={500} max={25000} step={500} onChange={v => updateParam('marketing_spend', v)} unit="$" />
          <SliderControl label="Number of Features" value={scenario.num_features} min={1} max={20} step={1} onChange={v => updateParam('num_features', v)} />
          <SliderControl label="Usage Level" value={scenario.usage} min={0} max={100} step={5} onChange={v => updateParam('usage', v)} unit="%" />
          <SliderControl label="Impressions" value={scenario.impressions} min={1000} max={100000} step={1000} onChange={v => updateParam('impressions', v)} />
          <SliderControl label="Clicks" value={scenario.clicks} min={50} max={10000} step={50} onChange={v => updateParam('clicks', v)} />
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Customer Sentiment Text</label>
            <textarea
              value={scenario.text}
              onChange={e => updateParam('text', e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              rows={2}
            />
          </div>
        </div>

        {/* Results */}
        <div className="lg:col-span-2 space-y-4">
          {results ? (
            <>
              {/* Metric Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <MetricCard
                  label="Churn Probability"
                  value={results.churn?.churn_probability}
                  format="percent"
                  color={results.churn?.risk_level === 'high' ? 'red' : results.churn?.risk_level === 'medium' ? 'yellow' : 'green'}
                  delta={comparison?.deltas?.churn?.churn_probability}
                />
                <MetricCard
                  label="Conversion Rate"
                  value={results.marketing?.conversion_rate}
                  format="percent"
                  color="blue"
                  delta={comparison?.deltas?.marketing?.conversion_rate}
                />
                <MetricCard
                  label="Demand"
                  value={results.pricing?.demand}
                  format="number"
                  color="green"
                  delta={comparison?.deltas?.pricing?.demand}
                />
                <MetricCard
                  label="Sentiment"
                  value={results.sentiment?.sentiment_score}
                  format="percent"
                  color={results.sentiment?.label === 'positive' ? 'green' : 'red'}
                  delta={comparison?.deltas?.sentiment?.sentiment_score}
                />
              </div>

              {/* Charts */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white rounded-xl p-4 border border-gray-200">
                  <h4 className="text-sm font-semibold text-gray-700 mb-3">Prediction Overview</h4>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={getBarData(results)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                      <YAxis tick={{ fontSize: 12 }} />
                      <Tooltip />
                      <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="bg-white rounded-xl p-4 border border-gray-200">
                  <h4 className="text-sm font-semibold text-gray-700 mb-3">Scenario vs Baseline</h4>
                  {comparison && (
                    <ResponsiveContainer width="100%" height={250}>
                      <RadarChart data={getRadarData(comparison)}>
                        <PolarGrid />
                        <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} />
                        <Radar name="Baseline" dataKey="baseline" stroke="#94a3b8" fill="#94a3b8" fillOpacity={0.2} />
                        <Radar name="Scenario" dataKey="scenario" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                        <Legend />
                      </RadarChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </div>

              {/* Propagation Trace */}
              {results.propagation_trace && (
                <div className="bg-white rounded-xl p-4 border border-gray-200">
                  <h4 className="text-sm font-semibold text-gray-700 mb-3">Causal Propagation Trace</h4>
                  <div className="flex flex-wrap gap-3">
                    {Object.entries(results.propagation_trace).map(([key, val]) => (
                      <div key={key} className="bg-gray-50 rounded-lg px-3 py-2">
                        <span className="text-xs text-gray-500">{key}</span>
                        <p className="text-sm font-medium text-gray-900">{typeof val === 'number' ? val.toFixed(4) : String(val)}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* SHAP Drivers */}
              {results.churn?.shap_drivers && (
                <div className="bg-white rounded-xl p-4 border border-gray-200">
                  <h4 className="text-sm font-semibold text-gray-700 mb-3">Top Churn Drivers (SHAP)</h4>
                  <div className="space-y-2">
                    {results.churn.shap_drivers.map((d, i) => (
                      <div key={i} className="flex items-center gap-3">
                        <span className="text-sm text-gray-600 w-36">{d.feature}</span>
                        <div className="flex-1 h-4 bg-gray-100 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${d.direction === 'positive' ? 'bg-red-400' : 'bg-green-400'}`}
                            style={{ width: `${Math.min(d.importance * 100, 100)}%` }}
                          />
                        </div>
                        <span className={`text-xs font-medium ${d.direction === 'positive' ? 'text-red-600' : 'text-green-600'}`}>
                          {d.direction === 'positive' ? '↑ churn' : '↓ churn'}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="bg-white rounded-xl p-12 border border-gray-200 text-center">
              <BarChart className="w-12 h-12 text-gray-300 mx-auto mb-4" />
              <p className="text-gray-500">Adjust parameters and click Simulate to see predictions</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function SliderControl({ label, value, min, max, step, onChange, unit }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; unit?: string
}) {
  return (
    <div>
      <div className="flex justify-between items-center mb-1">
        <label className="text-sm font-medium text-gray-700">{label}</label>
        <span className="text-sm font-semibold text-primary-600">{unit}{value.toLocaleString()}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
      />
    </div>
  )
}

function MetricCard({ label, value, format, color, delta }: {
  label: string; value?: number; format: 'percent' | 'number'; color: string; delta?: number
}) {
  const displayValue = value !== undefined
    ? format === 'percent' ? `${(value * 100).toFixed(1)}%` : value.toFixed(0)
    : '--'

  const colorMap: Record<string, string> = {
    red: 'border-red-200 bg-red-50',
    green: 'border-green-200 bg-green-50',
    blue: 'border-blue-200 bg-blue-50',
    yellow: 'border-yellow-200 bg-yellow-50',
  }

  return (
    <div className={`rounded-xl p-4 border ${colorMap[color] || 'border-gray-200 bg-white'}`}>
      <p className="text-xs text-gray-600 mb-1">{label}</p>
      <p className="text-xl font-bold text-gray-900">{displayValue}</p>
      {delta !== undefined && delta !== 0 && (
        <p className={`text-xs mt-1 font-medium ${delta > 0 ? 'text-red-600' : 'text-green-600'}`}>
          {delta > 0 ? '↑' : '↓'} {Math.abs(delta * 100).toFixed(2)}% vs baseline
        </p>
      )}
    </div>
  )
}

function getBarData(results: SimulationResult) {
  return [
    { name: 'Churn', value: (results.churn?.churn_probability || 0) * 100 },
    { name: 'Conversion', value: (results.marketing?.conversion_rate || 0) * 100 },
    { name: 'Sentiment', value: (results.sentiment?.sentiment_score || 0) * 100 },
    { name: 'Demand', value: results.pricing?.demand || 0 },
  ]
}

function getRadarData(comparison: any) {
  const b = comparison.baseline
  const s = comparison.scenario
  return [
    { metric: 'Churn', baseline: (b.churn?.churn_probability || 0) * 100, scenario: (s.churn?.churn_probability || 0) * 100 },
    { metric: 'Conversion', baseline: (b.marketing?.conversion_rate || 0) * 100, scenario: (s.marketing?.conversion_rate || 0) * 100 },
    { metric: 'Sentiment', baseline: (b.sentiment?.sentiment_score || 0) * 100, scenario: (s.sentiment?.sentiment_score || 0) * 100 },
    { metric: 'Demand', baseline: (b.pricing?.demand || 0) / 10, scenario: (s.pricing?.demand || 0) / 10 },
  ]
}

import { useState } from 'react'
import { chat } from '../api/client'
import { Send, Bot, User } from 'lucide-react'
import ReactMarkdown from 'react-markdown'

interface Message {
  role: 'user' | 'assistant'
  content: string
  data?: any
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: "I'm your AI business advisor. Ask me anything about your business scenarios. For example:\n\n- \"What happens if I increase price by 20%?\"\n- \"How can I reduce customer churn?\"\n- \"What's the impact of doubling marketing spend?\"\n\nI'll run the simulation, analyze the results with SHAP explanations, and provide counterfactual recommendations.",
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSend = async () => {
    if (!input.trim() || loading) return

    const userMsg = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: userMsg }])
    setLoading(true)

    try {
      const res = await chat(userMsg)
      const data = res.data

      let content = data.insight || 'No insight generated. Please train models first.'
      if (data.predictions?.churn?.churn_probability !== undefined) {
        content = `**Simulation Results:**\n- Churn: ${(data.predictions.churn.churn_probability * 100).toFixed(1)}% (${data.predictions.churn.risk_level})\n- Sentiment: ${(data.predictions.sentiment?.sentiment_score * 100).toFixed(1)}%\n- Conversion: ${(data.predictions.marketing?.conversion_rate * 100).toFixed(2)}%\n- Demand: ${data.predictions.pricing?.demand?.toFixed(0) || 'N/A'}\n\n---\n\n${content}`
      }

      setMessages(prev => [...prev, { role: 'assistant', content, data }])
    } catch (e: any) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${e.response?.data?.detail || 'Failed to process query. Make sure models are trained and GROQ_API_KEY is set.'}`,
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-12rem)]">
      <div className="mb-4">
        <h2 className="text-2xl font-bold text-gray-900">AI Business Chat</h2>
        <p className="text-gray-600">Ask natural language questions about business scenarios</p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 pb-4">
        {messages.map((msg, i) => (
          <div key={i} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}>
            {msg.role === 'assistant' && (
              <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center flex-shrink-0">
                <Bot size={16} className="text-primary-600" />
              </div>
            )}
            <div className={`max-w-[70%] rounded-xl px-4 py-3 ${
              msg.role === 'user'
                ? 'bg-primary-600 text-white'
                : 'bg-white border border-gray-200 text-gray-800'
            }`}>
              {msg.role === 'assistant' ? (
                <div className="prose prose-sm max-w-none">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
              ) : (
                <p className="text-sm">{msg.content}</p>
              )}
            </div>
            {msg.role === 'user' && (
              <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center flex-shrink-0">
                <User size={16} className="text-gray-600" />
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="flex gap-3">
            <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
              <Bot size={16} className="text-primary-600" />
            </div>
            <div className="bg-white border border-gray-200 rounded-xl px-4 py-3">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="flex gap-3 pt-4 border-t border-gray-200">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          placeholder="Ask about business scenarios... (e.g., 'What if price increases by 30%?')"
          className="flex-1 border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          disabled={loading}
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          className="px-4 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 transition-colors"
        >
          <Send size={20} />
        </button>
      </div>
    </div>
  )
}

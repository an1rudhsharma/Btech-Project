import { useState, useRef, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Settings, Plus, MessageSquare, Paperclip, Send, Bot, User, X, Trash2, ChevronDown } from 'lucide-react'
import { getStatus, trainAll, uploadDataset, chat as chatApi } from './api/client'
import ReactMarkdown from 'react-markdown'
import toast, { Toaster } from 'react-hot-toast'

interface Message {
  role: 'user' | 'assistant'
  content: string
  file?: string
}

interface ChatSession {
  id: string
  title: string
  messages: Message[]
}

function App() {
  const [sessions, setSessions] = useState<ChatSession[]>([
    { id: '1', title: 'New Chat', messages: [] }
  ])
  const [activeSessionId, setActiveSessionId] = useState('1')
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const { data: statusData, refetch: refetchStatus } = useQuery({
    queryKey: ['status'],
    queryFn: () => getStatus().then(r => r.data),
    refetchInterval: 30000,
  })

  const activeSession = sessions.find(s => s.id === activeSessionId)!
  const models = statusData?.models || {}

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [activeSession.messages])

  const updateSession = (id: string, updates: Partial<ChatSession>) => {
    setSessions(prev => prev.map(s => s.id === id ? { ...s, ...updates } : s))
  }

  const newChat = () => {
    const id = Date.now().toString()
    setSessions(prev => [...prev, { id, title: 'New Chat', messages: [] }])
    setActiveSessionId(id)
  }

  const deleteSession = (id: string) => {
    if (sessions.length === 1) {
      updateSession(id, { messages: [], title: 'New Chat' })
      return
    }
    setSessions(prev => prev.filter(s => s.id !== id))
    if (activeSessionId === id) {
      setActiveSessionId(sessions.find(s => s.id !== id)!.id)
    }
  }

  const handleFileUpload = async (file: File) => {
    const userMsg: Message = { role: 'user', content: `Uploading dataset: **${file.name}**`, file: file.name }
    updateSession(activeSessionId, { messages: [...activeSession.messages, userMsg] })

    try {
      const res = await uploadDataset(file)
      const data = res.data
      const assistantMsg: Message = {
        role: 'assistant',
        content: `Dataset **${data.filename}** uploaded successfully!\n\n- **Rows:** ${data.rows}\n- **Columns:** ${data.columns?.length}\n- **Detected mappings:** ${Object.entries(data.detected_mapping || {}).filter(([, v]) => v).map(([k, v]) => `${k} → ${v}`).join(', ') || 'None'}\n\nYou can now ask me to train models on this data or run simulations. Try:\n- "Train all models"\n- "What would happen if price increases by 30%?"\n- "How can I reduce churn?"`,
      }
      updateSession(activeSessionId, { messages: [...activeSession.messages, userMsg, assistantMsg] })
      if (activeSession.title === 'New Chat') {
        updateSession(activeSessionId, { title: file.name.replace(/\.[^.]+$/, '') })
      }
    } catch (e: any) {
      const errMsg: Message = { role: 'assistant', content: `Upload failed: ${e.response?.data?.detail || e.message}` }
      updateSession(activeSessionId, { messages: [...activeSession.messages, userMsg, errMsg] })
    }
  }

  const handleSend = async () => {
    if (!input.trim() || loading) return
    const userText = input.trim()
    setInput('')

    const userMsg: Message = { role: 'user', content: userText }
    const updatedMessages = [...activeSession.messages, userMsg]
    updateSession(activeSessionId, { messages: updatedMessages })

    if (activeSession.title === 'New Chat' && activeSession.messages.length === 0) {
      updateSession(activeSessionId, { title: userText.slice(0, 40) })
    }

    const trainKeywords = /^(train all|train models|train everything)/i
    if (trainKeywords.test(userText)) {
      setLoading(true)
      try {
        await trainAll()
        refetchStatus()
        const assistantMsg: Message = { role: 'assistant', content: 'All models trained successfully! You can now run simulations and ask business questions.' }
        updateSession(activeSessionId, { messages: [...updatedMessages, assistantMsg] })
      } catch (e: any) {
        const errMsg: Message = { role: 'assistant', content: `Training failed: ${e.response?.data?.detail || e.message}` }
        updateSession(activeSessionId, { messages: [...updatedMessages, errMsg] })
      } finally {
        setLoading(false)
      }
      return
    }

    setLoading(true)
    try {
      const res = await chatApi(userText)
      const data = res.data
      let content = data.insight || 'No insight generated. Please train models first.'
      if (data.predictions?.churn?.churn_probability !== undefined) {
        const p = data.predictions
        content = `### Simulation Results\n\n| Metric | Value |\n|--------|-------|\n| Churn Risk | ${(p.churn.churn_probability * 100).toFixed(1)}% (${p.churn.risk_level}) |\n| Sentiment | ${(p.sentiment?.sentiment_score * 100).toFixed(1)}% (${p.sentiment?.label}) |\n| Conversion | ${(p.marketing?.conversion_rate * 100).toFixed(2)}% |\n| Demand | ${p.pricing?.demand?.toFixed(0) || 'N/A'} |\n\n---\n\n${content}`
      }
      if (data.counterfactuals?.length > 0) {
        content += '\n\n### Recommended Changes\n'
        data.counterfactuals.forEach((cf: any, i: number) => {
          content += `\n**Option ${i + 1}** (feasibility: ${(cf.feasibility * 100).toFixed(0)}%)\n`
          cf.changes?.forEach((c: any) => {
            content += `- ${c.feature}: ${c.from} → ${c.to} (${c.change_pct > 0 ? '+' : ''}${c.change_pct}%)\n`
          })
        })
      }
      const assistantMsg: Message = { role: 'assistant', content }
      updateSession(activeSessionId, { messages: [...updatedMessages, assistantMsg] })
    } catch (e: any) {
      const errMsg: Message = {
        role: 'assistant',
        content: `Error: ${e.response?.data?.detail || 'Failed to process. Make sure models are trained and GROQ_API_KEY is set.'}`,
      }
      updateSession(activeSessionId, { messages: [...updatedMessages, errMsg] })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-screen flex bg-[#f9fafb]">
      {/* Sidebar */}
      <aside className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-200 overflow-hidden flex-shrink-0 bg-white border-r border-border-subtle flex flex-col`}>
        <div className="p-3">
          <button
            onClick={newChat}
            className="w-full flex items-center gap-2 px-3 py-2.5 border border-border-subtle rounded-lg text-sm text-text-primary hover:bg-surface-hover transition-all"
          >
            <Plus size={16} /> New Chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-2 space-y-0.5">
          {sessions.map(session => (
            <div
              key={session.id}
              className={`group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer text-sm transition-all ${
                session.id === activeSessionId ? 'bg-accent-dim text-accent font-medium' : 'text-text-secondary hover:bg-surface-hover'
              }`}
              onClick={() => setActiveSessionId(session.id)}
            >
              <MessageSquare size={14} className="flex-shrink-0" />
              <span className="truncate flex-1">{session.title}</span>
              <button
                onClick={(e) => { e.stopPropagation(); deleteSession(session.id) }}
                className="opacity-0 group-hover:opacity-100 text-text-muted hover:text-danger transition-all"
              >
                <Trash2 size={12} />
              </button>
            </div>
          ))}
        </div>

        {/* Settings panel */}
        <div className="border-t border-border-subtle p-3">
          <button
            onClick={() => setSettingsOpen(!settingsOpen)}
            className="w-full flex items-center gap-2 px-3 py-2 text-sm text-text-secondary hover:text-text-primary hover:bg-surface-hover rounded-lg transition-all"
          >
            <Settings size={14} />
            <span>Settings</span>
            <ChevronDown size={12} className={`ml-auto transition-transform ${settingsOpen ? 'rotate-180' : ''}`} />
          </button>
          {settingsOpen && (
            <div className="mt-2 p-3 bg-surface-hover rounded-lg space-y-3">
              <div>
                <p className="text-[11px] font-medium text-text-muted uppercase tracking-wider mb-2">Model Status</p>
                <div className="space-y-1.5">
                  {Object.entries(models).map(([name, info]: [string, any]) => (
                    <div key={name} className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${info?.trained ? 'bg-success' : 'bg-gray-300'}`} />
                      <span className="text-xs text-text-secondary capitalize">{name}</span>
                      {info?.trained && <span className="text-[10px] text-success ml-auto">Ready</span>}
                    </div>
                  ))}
                </div>
              </div>
              <p className="text-[10px] text-text-muted">Type "train all" in chat to train models on sample data.</p>
            </div>
          )}
        </div>
      </aside>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <header className="h-12 flex items-center px-4 border-b border-border-subtle bg-white">
          <button onClick={() => setSidebarOpen(!sidebarOpen)} className="text-text-secondary hover:text-text-primary mr-3">
            <MessageSquare size={18} />
          </button>
          <span className="text-sm font-medium text-text-primary truncate">AI Business Decision Simulator</span>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
            {activeSession.messages.length === 0 && (
              <div className="text-center py-20">
                <div className="w-14 h-14 rounded-full bg-accent-dim flex items-center justify-center mx-auto mb-4">
                  <Bot size={28} className="text-accent" />
                </div>
                <h2 className="text-xl font-semibold text-text-primary mb-2">AI Business Simulator</h2>
                <p className="text-text-secondary text-sm max-w-md mx-auto mb-8">
                  Ask questions about pricing, churn, marketing, or sentiment. Upload data, train models, and get AI-powered business insights — all through conversation.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-lg mx-auto">
                  {[
                    'What happens if I raise prices by 25%?',
                    'How can I reduce customer churn?',
                    'Train all models',
                    'What drives conversion rate?',
                  ].map(suggestion => (
                    <button
                      key={suggestion}
                      onClick={() => { setInput(suggestion) }}
                      className="text-left px-4 py-3 bg-white border border-border-subtle rounded-lg text-sm text-text-secondary hover:border-accent/40 hover:text-text-primary transition-all"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {activeSession.messages.map((msg, i) => (
              <div key={i} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}>
                {msg.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-full bg-accent-dim flex items-center justify-center flex-shrink-0">
                    <Bot size={16} className="text-accent" />
                  </div>
                )}
                <div className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                  msg.role === 'user'
                    ? 'bg-accent text-white'
                    : 'bg-white border border-border-subtle shadow-sm'
                }`}>
                  {msg.role === 'assistant' ? (
                    <div className="prose prose-sm max-w-none text-text-primary [&_strong]:text-text-primary [&_p]:text-text-secondary [&_li]:text-text-secondary [&_code]:text-accent [&_hr]:border-border-subtle [&_table]:text-xs [&_th]:text-text-muted [&_td]:text-text-primary [&_h3]:text-text-primary [&_h3]:text-sm [&_h3]:mt-4 [&_h3]:mb-2">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                  ) : (
                    <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                  )}
                </div>
                {msg.role === 'user' && (
                  <div className="w-8 h-8 rounded-full bg-surface-hover flex items-center justify-center flex-shrink-0 border border-border-subtle">
                    <User size={16} className="text-text-secondary" />
                  </div>
                )}
              </div>
            ))}

            {loading && (
              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-accent-dim flex items-center justify-center">
                  <Bot size={16} className="text-accent" />
                </div>
                <div className="bg-white border border-border-subtle rounded-2xl px-4 py-3 shadow-sm">
                  <div className="flex gap-1.5">
                    <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input bar */}
        <div className="border-t border-border-subtle bg-white p-4">
          <div className="max-w-3xl mx-auto">
            <div className="flex items-end gap-2 bg-surface-hover border border-border-subtle rounded-xl px-3 py-2 focus-within:border-accent/40 focus-within:ring-1 focus-within:ring-accent/10 transition-all">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="p-1.5 text-text-muted hover:text-text-primary transition-colors"
                title="Upload dataset"
              >
                <Paperclip size={18} />
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.xlsx"
                className="hidden"
                onChange={e => { if (e.target.files?.[0]) handleFileUpload(e.target.files[0]); e.target.value = '' }}
              />
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() } }}
                placeholder="Ask anything about your business data..."
                className="flex-1 bg-transparent text-sm text-text-primary placeholder-text-muted resize-none focus:outline-none min-h-[24px] max-h-[120px] py-1"
                rows={1}
                disabled={loading}
              />
              <button
                onClick={handleSend}
                disabled={loading || !input.trim()}
                className="p-1.5 text-accent hover:text-accent-strong disabled:text-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                <Send size={18} />
              </button>
            </div>
            <p className="text-[10px] text-text-muted text-center mt-2">
              Upload CSV/Excel files or ask about pricing, churn, marketing & sentiment.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

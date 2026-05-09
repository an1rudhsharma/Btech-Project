import { useState, useRef, useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { Settings, Plus, MessageSquare, Send, Bot, User, Trash2, ChevronDown, BookOpen, LogOut, Database } from 'lucide-react'
import { getStatus, trainAll, sessionsApi, streamChat } from '../api/client'
import { useAuth } from '../contexts/AuthContext'
import ReactMarkdown from 'react-markdown'
import toast from 'react-hot-toast'

interface Message {
  id?: string
  role: 'user' | 'assistant'
  content: string
  streaming?: boolean
}

interface ChatSession {
  id: string
  title: string
  created_at?: string
  updated_at?: string
}

function getSuggestions(models: Record<string, any>): string[] {
  const anyTrained = Object.values(models).some((m: any) => m?.trained)
  if (!anyTrained) {
    return ['Train on sample data', 'What can you do?', 'What happens if I raise prices by 25%?']
  }
  return [
    'What happens if I raise price by 25%?',
    'How can I reduce customer churn?',
    'What drives conversion rate?',
    'What if marketing spend doubles?',
    'Analyze sentiment distribution',
  ]
}

export default function Chat() {
  const { user, signOut } = useAuth()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const { data: statusData, refetch: refetchStatus } = useQuery({
    queryKey: ['status'],
    queryFn: () => getStatus().then(r => r.data),
    refetchInterval: 30000,
  })

  const models = statusData?.models || {}
  const activeSession = sessions.find(s => s.id === activeSessionId)
  const suggestions = getSuggestions(models)

  useEffect(() => {
    loadSessions()
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const loadSessions = async () => {
    try {
      const res = await sessionsApi.list()
      const data = res.data || []
      setSessions(data)
      if (data.length > 0 && !activeSessionId) {
        setActiveSessionId(data[0].id)
        loadMessages(data[0].id)
      } else if (data.length === 0) {
        await newChat()
      }
    } catch {
      const localSession: ChatSession = { id: 'local-1', title: 'New Chat' }
      setSessions([localSession])
      setActiveSessionId('local-1')
    }
  }

  const loadMessages = async (sessionId: string) => {
    try {
      const res = await sessionsApi.messages(sessionId)
      setMessages((res.data || []).map((m: any) => ({
        id: m.id,
        role: m.role,
        content: m.content,
      })))
    } catch {
      setMessages([])
    }
  }

  const newChat = async () => {
    try {
      const res = await sessionsApi.create('New Chat')
      const session = res.data
      setSessions(prev => [session, ...prev])
      setActiveSessionId(session.id)
      setMessages([])
    } catch {
      const id = Date.now().toString()
      setSessions(prev => [{ id, title: 'New Chat' }, ...prev])
      setActiveSessionId(id)
      setMessages([])
    }
  }

  const switchSession = (sessionId: string) => {
    setActiveSessionId(sessionId)
    loadMessages(sessionId)
  }

  const deleteSession = async (id: string) => {
    try {
      await sessionsApi.delete(id)
    } catch {}
    setSessions(prev => prev.filter(s => s.id !== id))
    if (activeSessionId === id) {
      const remaining = sessions.filter(s => s.id !== id)
      if (remaining.length > 0) {
        switchSession(remaining[0].id)
      } else {
        newChat()
      }
    }
  }

  const handleSend = async (overrideText?: string) => {
    const userText = (overrideText || input).trim()
    if (!userText) return
    if (loading) return

    setInput('')
    await processTextMessage(userText)
  }

  const processTextMessage = async (userText: string) => {
    setMessages(prev => [...prev, { role: 'user', content: userText }])

    if (activeSession?.title === 'New Chat' && activeSessionId) {
      const title = userText.slice(0, 50)
      setSessions(prev => prev.map(s => s.id === activeSessionId ? { ...s, title } : s))
      sessionsApi.update(activeSessionId, { title }).catch(() => {})
    }

    // Handle train command locally
    const trainKeywords = /^(train\s*(all|the|my|every)?\s*(model|models|everything|data|on sample data)?|train\s*$)/i
    if (trainKeywords.test(userText)) {
      setLoading(true)
      try {
        await trainAll()
        refetchStatus()
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: 'All models trained successfully! You can now run simulations and ask business questions.\n\nTry asking:\n- "What happens if I raise prices by 25%?"\n- "How can I reduce churn?"\n- "What drives conversion rate?"'
        }])
      } catch (e: any) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `Training failed: ${e.response?.data?.detail || e.message}`
        }])
      } finally {
        setLoading(false)
      }
      return
    }

    // Stream chat response
    setLoading(true)
    setMessages(prev => [...prev, { role: 'assistant', content: '', streaming: true }])

    try {
      if (!activeSessionId) throw new Error('No active session')
      const stream = await streamChat(userText, activeSessionId)
      const reader = stream.getReader()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        setMessages(prev => {
          const last = prev[prev.length - 1]
          if (last.streaming) {
            return [...prev.slice(0, -1), { ...last, content: last.content + value }]
          }
          return prev
        })
      }

      setMessages(prev => {
        const last = prev[prev.length - 1]
        return [...prev.slice(0, -1), { ...last, streaming: false }]
      })
    } catch (e: any) {
      setMessages(prev => {
        const last = prev[prev.length - 1]
        if (last.streaming) {
          return [...prev.slice(0, -1), {
            role: 'assistant' as const,
            content: `Error: ${e.message || 'Failed to process. Check that models are trained and GROQ_API_KEY is set.'}`,
            streaming: false,
          }]
        }
        return [...prev, { role: 'assistant', content: `Error: ${e.message}` }]
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-screen flex bg-[#f9fafb]">
      {/* Sidebar */}
      <aside className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-200 overflow-hidden flex-shrink-0 bg-white border-r border-gray-100 flex flex-col`}>
        <div className="p-3">
          <button
            onClick={newChat}
            className="w-full flex items-center gap-2 px-3 py-2.5 border border-gray-200 rounded-lg text-sm text-gray-700 hover:bg-gray-50 transition-all"
          >
            <Plus size={16} /> New Chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-2 space-y-0.5">
          {sessions.map(session => (
            <div
              key={session.id}
              className={`group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer text-sm transition-all ${
                session.id === activeSessionId ? 'bg-blue-50 text-blue-700 font-medium' : 'text-gray-600 hover:bg-gray-50'
              }`}
              onClick={() => switchSession(session.id)}
            >
              <MessageSquare size={14} className="flex-shrink-0" />
              <span className="truncate flex-1">{session.title}</span>
              <button
                onClick={(e) => { e.stopPropagation(); deleteSession(session.id) }}
                className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-500 transition-all"
              >
                <Trash2 size={12} />
              </button>
            </div>
          ))}
        </div>

        {/* Bottom section */}
        <div className="border-t border-gray-100 p-3 space-y-1">
          <button
            onClick={() => navigate('/knowledge')}
            className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-50 rounded-lg transition-all"
          >
            <Database size={14} /> Knowledge Center
          </button>
          <button
            onClick={() => setSettingsOpen(!settingsOpen)}
            className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-50 rounded-lg transition-all"
          >
            <Settings size={14} />
            <span>Settings</span>
            <ChevronDown size={12} className={`ml-auto transition-transform ${settingsOpen ? 'rotate-180' : ''}`} />
          </button>
          {settingsOpen && (
            <div className="p-3 bg-gray-50 rounded-lg space-y-3">
              <div>
                <p className="text-[11px] font-medium text-gray-400 uppercase tracking-wider mb-2">Model Status</p>
                <div className="space-y-1.5">
                  {Object.entries(models).map(([name, info]: [string, any]) => (
                    <div key={name} className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${info?.trained ? 'bg-green-500' : 'bg-gray-300'}`} />
                      <span className="text-xs text-gray-600 capitalize">{name}</span>
                      {info?.trained && <span className="text-[10px] text-green-600 ml-auto">Ready</span>}
                    </div>
                  ))}
                </div>
              </div>
              <p className="text-[10px] text-gray-400">Logged in as {user?.email}</p>
            </div>
          )}
          <button
            onClick={signOut}
            className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:text-red-600 hover:bg-red-50 rounded-lg transition-all"
          >
            <LogOut size={14} /> Sign Out
          </button>
        </div>
      </aside>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-12 flex items-center px-4 border-b border-gray-100 bg-white">
          <button onClick={() => setSidebarOpen(!sidebarOpen)} className="text-gray-500 hover:text-gray-700 mr-3">
            <MessageSquare size={18} />
          </button>
          <span className="text-sm font-medium text-gray-800 truncate">
            {activeSession?.title || 'AI Business Simulator'}
          </span>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
            {messages.length === 0 && (
              <div className="text-center py-20">
                <div className="w-14 h-14 rounded-full bg-blue-50 flex items-center justify-center mx-auto mb-4">
                  <Bot size={28} className="text-blue-600" />
                </div>
                <h2 className="text-xl font-semibold text-gray-900 mb-2">AI Business Simulator</h2>
                <p className="text-gray-500 text-sm max-w-md mx-auto mb-4">
                  Ask AI-powered business questions about pricing, churn, marketing & sentiment.
                </p>
                <button
                  onClick={() => navigate('/knowledge')}
                  className="inline-flex items-center gap-2 px-4 py-2 mb-8 bg-blue-50 border border-blue-100 rounded-lg text-sm text-blue-700 hover:bg-blue-100 transition-all"
                >
                  <Database size={16} />
                  Upload datasets in Knowledge Center to auto-train models
                </button>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-lg mx-auto">
                  {suggestions.map(suggestion => (
                    <button
                      key={suggestion}
                      onClick={() => handleSend(suggestion)}
                      className="text-left px-4 py-3 bg-white border border-gray-200 rounded-lg text-sm text-gray-600 hover:border-blue-300 hover:text-gray-800 transition-all"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg, i) => (
              <div key={i}>
                <div className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}>
                  {msg.role === 'assistant' && (
                    <div className="w-8 h-8 rounded-full bg-blue-50 flex items-center justify-center flex-shrink-0">
                      <Bot size={16} className="text-blue-600" />
                    </div>
                  )}
                  <div className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                    msg.role === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-white border border-gray-100 shadow-sm'
                  }`}>
                    {msg.role === 'assistant' ? (
                      <div className="prose prose-sm max-w-none text-gray-800 [&_table]:text-xs">
                        <ReactMarkdown>{msg.content || '\u200B'}</ReactMarkdown>
                        {msg.streaming && (
                          <span className="inline-block w-1.5 h-4 bg-blue-500 animate-pulse ml-0.5 -mb-0.5" />
                        )}
                      </div>
                    ) : (
                      <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                    )}
                  </div>
                  {msg.role === 'user' && (
                    <div className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center flex-shrink-0">
                      <User size={16} className="text-gray-600" />
                    </div>
                  )}
                </div>

                {msg.role === 'assistant' && i === messages.length - 1 && !loading && !msg.streaming && (
                  <div className="ml-11 mt-3 flex flex-wrap gap-2">
                    {suggestions.slice(0, 3).map(s => (
                      <button
                        key={s}
                        onClick={() => handleSend(s)}
                        className="px-3 py-1.5 bg-gray-50 border border-gray-200 rounded-full text-xs text-gray-500 hover:text-gray-800 hover:border-blue-300 transition-all"
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ))}

            {loading && messages[messages.length - 1]?.role !== 'assistant' && (
              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-blue-50 flex items-center justify-center">
                  <Bot size={16} className="text-blue-600" />
                </div>
                <div className="bg-white border border-gray-100 rounded-2xl px-4 py-3 shadow-sm">
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
        <div className="border-t border-gray-100 bg-white p-4">
          <div className="max-w-3xl mx-auto">
            <div className="flex items-end gap-2 bg-gray-50 border border-gray-200 rounded-xl px-3 py-2 focus-within:border-blue-300 focus-within:ring-1 focus-within:ring-blue-100 transition-all">
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() } }}
                placeholder="Ask anything about your business data..."
                className="flex-1 bg-transparent text-sm text-gray-800 placeholder-gray-400 resize-none focus:outline-none min-h-[24px] max-h-[120px] py-1"
                rows={1}
                disabled={loading}
              />
              <button
                onClick={() => handleSend()}
                disabled={loading || !input.trim()}
                className="p-1.5 text-blue-600 hover:text-blue-700 disabled:text-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                <Send size={18} />
              </button>
            </div>
            <p className="text-[10px] text-gray-400 text-center mt-2">
              Upload datasets in the Knowledge Center to auto-train ML models.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

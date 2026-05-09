import { useState, useEffect } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { LogIn, UserPlus, Brain, ArrowLeft } from 'lucide-react'
import toast from 'react-hot-toast'

export default function Login() {
  const [searchParams] = useSearchParams()
  const [mode, setMode] = useState<'signin' | 'signup' | 'forgot'>('signin')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const { signIn, signUp, resetPassword } = useAuth()
  const navigate = useNavigate()

  useEffect(() => {
    if (searchParams.get('signup') === 'true') {
      setMode('signup')
    }
  }, [searchParams])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!email) return
    if (mode !== 'forgot' && !password) return

    setLoading(true)
    try {
      if (mode === 'forgot') {
        const { error } = await resetPassword(email)
        if (error) {
          toast.error(error.message)
        } else {
          toast.success('Password reset link sent! Check your email.')
          setMode('signin')
        }
      } else {
        const { error } = mode === 'signup'
          ? await signUp(email, password)
          : await signIn(email, password)

        if (error) {
          toast.error(error.message)
        } else {
          if (mode === 'signup') {
            toast.success('Account created! Check your email to confirm.')
          } else {
            navigate('/chat')
          }
        }
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="w-full max-w-md mx-4">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-blue-600 text-white mb-4">
            <Brain size={32} />
          </div>
          <h1 className="text-2xl font-bold text-gray-900">AI Business Simulator</h1>
          <p className="text-sm text-gray-500 mt-1">
            Simulate pricing, churn, marketing & sentiment decisions
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-xl border border-gray-100 p-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-6">
            {mode === 'signup' ? 'Create an account' : mode === 'forgot' ? 'Reset your password' : 'Welcome back'}
          </h2>

          {mode === 'forgot' && (
            <p className="text-sm text-gray-500 mb-4">
              Enter your email and we'll send you a link to reset your password.
            </p>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full px-4 py-2.5 rounded-lg border border-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all"
                required
              />
            </div>
            {mode !== 'forgot' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
                <input
                  type="password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder=""
                  className="w-full px-4 py-2.5 rounded-lg border border-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all"
                  required
                  minLength={6}
                />
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white text-sm font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              {loading ? (
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : mode === 'signup' ? (
                <><UserPlus size={16} /> Create Account</>
              ) : mode === 'forgot' ? (
                <>Send Reset Link</>
              ) : (
                <><LogIn size={16} /> Sign In</>
              )}
            </button>
          </form>

          {mode === 'signin' && (
            <div className="mt-4 text-center">
              <button
                onClick={() => setMode('forgot')}
                className="text-sm text-gray-500 hover:text-blue-600 transition-colors"
              >
                Forgot your password?
              </button>
            </div>
          )}

          <div className="mt-4 text-center">
            {mode === 'forgot' ? (
              <button
                onClick={() => setMode('signin')}
                className="text-sm text-blue-600 hover:text-blue-700 font-medium inline-flex items-center gap-1"
              >
                <ArrowLeft size={14} /> Back to Sign In
              </button>
            ) : (
              <button
                onClick={() => setMode(mode === 'signup' ? 'signin' : 'signup')}
                className="text-sm text-blue-600 hover:text-blue-700 font-medium"
              >
                {mode === 'signup' ? 'Already have an account? Sign in' : "Don't have an account? Sign up"}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

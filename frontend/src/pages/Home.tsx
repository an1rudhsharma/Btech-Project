import { useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { motion, useScroll, useTransform } from 'framer-motion'
import { useAuth } from '../contexts/AuthContext'
import {
  Brain,
  TrendingUp,
  MessageSquare,
  DollarSign,
  Users,
  Upload,
  MessageCircle,
  Lightbulb,
  Zap,
  Database,
  Radio,
  Shield,
  ArrowRight,
  ChevronDown
} from 'lucide-react'

const fadeUp = {
  hidden: { opacity: 0, y: 40 },
  visible: { opacity: 1, y: 0 }
}

const staggerContainer = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.15 } }
}

const scaleIn = {
  hidden: { opacity: 0, scale: 0.8 },
  visible: { opacity: 1, scale: 1 }
}

const features = [
  {
    icon: Users,
    title: 'Churn Prediction',
    description: 'Identify at-risk customers before they leave using advanced ML models',
    gradient: 'from-red-500 to-orange-500'
  },
  {
    icon: MessageSquare,
    title: 'Sentiment Analysis',
    description: 'Understand customer emotions from text using DistilBERT NLP',
    gradient: 'from-purple-500 to-pink-500'
  },
  {
    icon: TrendingUp,
    title: 'Marketing Impact',
    description: 'Simulate campaign ROI and predict conversion rates',
    gradient: 'from-blue-500 to-cyan-500'
  },
  {
    icon: DollarSign,
    title: 'Pricing Strategy',
    description: 'Optimize pricing with demand elasticity modeling',
    gradient: 'from-green-500 to-emerald-500'
  }
]

const steps = [
  { icon: Upload, title: 'Upload Data', description: 'Drop your CSV, Excel, or any dataset' },
  { icon: MessageCircle, title: 'Ask Questions', description: 'Chat naturally about your business' },
  { icon: Lightbulb, title: 'Get Insights', description: 'Receive AI-powered recommendations' }
]

const stats = [
  { icon: Brain, value: '4', label: 'ML Models' },
  { icon: Database, value: 'RAG', label: 'Powered Knowledge' },
  { icon: Radio, value: 'Live', label: 'Streaming Responses' },
  { icon: Shield, value: 'Secure', label: 'User Isolation' }
]

function FloatingCard({ children, delay, x, y }: { children: React.ReactNode; delay: number; x: number; y: number }) {
  return (
    <motion.div
      className="absolute"
      style={{ left: `${x}%`, top: `${y}%` }}
      animate={{
        y: [0, -15, 0],
        rotate: [0, 2, -2, 0]
      }}
      transition={{
        duration: 4,
        repeat: Infinity,
        delay,
        ease: 'easeInOut'
      }}
    >
      {children}
    </motion.div>
  )
}

function AnimatedCounter({ value, label, icon: Icon }: { value: string; label: string; icon: any }) {
  return (
    <motion.div
      variants={fadeUp}
      className="text-center"
    >
      <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-white/10 backdrop-blur-sm mb-3">
        <Icon size={24} className="text-blue-300" />
      </div>
      <div className="text-3xl font-bold text-white mb-1">{value}</div>
      <div className="text-sm text-blue-200">{label}</div>
    </motion.div>
  )
}

export default function Home() {
  const { user, loading } = useAuth()
  const navigate = useNavigate()
  const { scrollYProgress } = useScroll()
  const heroOpacity = useTransform(scrollYProgress, [0, 0.15], [1, 0])
  const heroScale = useTransform(scrollYProgress, [0, 0.15], [1, 0.95])

  useEffect(() => {
    if (!loading && user) {
      navigate('/chat', { replace: true })
    }
  }, [user, loading, navigate])

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-slate-950">
        <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  return (
    <div className="bg-slate-950 text-white overflow-hidden">
      {/* Navbar */}
      <motion.nav
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="fixed top-0 left-0 right-0 z-50 px-6 py-4"
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center">
              <Brain size={20} />
            </div>
            <span className="font-bold text-lg">BizSimAI</span>
          </div>
          <div className="flex items-center gap-3">
            <Link
              to="/login"
              className="px-4 py-2 text-sm font-medium text-gray-300 hover:text-white transition-colors"
            >
              Sign In
            </Link>
            <Link
              to="/login?signup=true"
              className="px-4 py-2 text-sm font-medium bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
            >
              Get Started
            </Link>
          </div>
        </div>
      </motion.nav>

      {/* Hero Section */}
      <motion.section
        style={{ opacity: heroOpacity, scale: heroScale }}
        className="relative min-h-screen flex items-center justify-center px-6"
      >
        {/* Animated gradient background */}
        <div className="absolute inset-0">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-600/20 via-purple-600/10 to-slate-950" />
          <motion.div
            className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-600/20 rounded-full blur-3xl"
            animate={{ x: [0, 50, 0], y: [0, -30, 0] }}
            transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
          />
          <motion.div
            className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-600/20 rounded-full blur-3xl"
            animate={{ x: [0, -50, 0], y: [0, 30, 0] }}
            transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut' }}
          />
        </div>

        {/* Floating model cards */}
        <div className="absolute inset-0 hidden lg:block">
          <FloatingCard delay={0} x={10} y={25}>
            <div className="px-4 py-3 bg-white/5 backdrop-blur-md border border-white/10 rounded-xl">
              <Users size={20} className="text-red-400" />
            </div>
          </FloatingCard>
          <FloatingCard delay={1} x={80} y={20}>
            <div className="px-4 py-3 bg-white/5 backdrop-blur-md border border-white/10 rounded-xl">
              <MessageSquare size={20} className="text-purple-400" />
            </div>
          </FloatingCard>
          <FloatingCard delay={0.5} x={85} y={60}>
            <div className="px-4 py-3 bg-white/5 backdrop-blur-md border border-white/10 rounded-xl">
              <TrendingUp size={20} className="text-blue-400" />
            </div>
          </FloatingCard>
          <FloatingCard delay={1.5} x={8} y={65}>
            <div className="px-4 py-3 bg-white/5 backdrop-blur-md border border-white/10 rounded-xl">
              <DollarSign size={20} className="text-green-400" />
            </div>
          </FloatingCard>
        </div>

        {/* Hero content */}
        <div className="relative z-10 text-center max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.2 }}
          >
            <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-blue-500/10 border border-blue-500/20 rounded-full text-sm text-blue-300 mb-6">
              <Zap size={14} />
              Powered by LLMs + Machine Learning
            </div>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.4 }}
            className="text-5xl md:text-7xl font-bold leading-tight mb-6"
          >
            Simulate Business{' '}
            <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              Decisions
            </span>{' '}
            with AI
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.6 }}
            className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto mb-10"
          >
            Upload your data, ask questions in plain English, and get ML-powered insights
            on pricing, churn, marketing, and sentiment — all in one chat interface.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.8 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Link
              to="/login?signup=true"
              className="group px-8 py-3.5 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-xl transition-all hover:shadow-lg hover:shadow-blue-600/25 flex items-center gap-2"
            >
              Get Started Free
              <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              to="/login"
              className="px-8 py-3.5 bg-white/5 hover:bg-white/10 border border-white/10 text-white font-medium rounded-xl transition-all"
            >
              Sign In
            </Link>
          </motion.div>
        </div>

        {/* Scroll indicator */}
        <motion.div
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <ChevronDown size={24} className="text-gray-500" />
        </motion.div>
      </motion.section>

      {/* Features Section */}
      <section className="relative py-32 px-6">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: '-100px' }}
            variants={fadeUp}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-5xl font-bold mb-4">
              Four Models.{' '}
              <span className="text-gray-500">One Interface.</span>
            </h2>
            <p className="text-gray-400 text-lg max-w-xl mx-auto">
              Each model works independently or in a causal loop — change one variable and watch the ripple effects.
            </p>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: '-50px' }}
            variants={staggerContainer}
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            {features.map((feature) => (
              <motion.div
                key={feature.title}
                variants={fadeUp}
                whileHover={{ scale: 1.02, y: -4 }}
                transition={{ type: 'spring', stiffness: 300 }}
                className="group relative p-8 rounded-2xl bg-white/[0.03] border border-white/[0.06] hover:border-white/[0.12] transition-all cursor-default"
              >
                <div className={`inline-flex p-3 rounded-xl bg-gradient-to-br ${feature.gradient} mb-4`}>
                  <feature.icon size={24} className="text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-gray-400">{feature.description}</p>
                <div className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${feature.gradient} opacity-0 group-hover:opacity-[0.03] transition-opacity`} />
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="relative py-32 px-6">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: '-100px' }}
            variants={fadeUp}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-5xl font-bold mb-4">
              How It{' '}
              <span className="text-gray-500">Works</span>
            </h2>
            <p className="text-gray-400 text-lg">Three steps to smarter decisions</p>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: '-50px' }}
            variants={staggerContainer}
            className="relative grid grid-cols-1 md:grid-cols-3 gap-8"
          >
            {/* Connecting line */}
            <div className="hidden md:block absolute top-1/2 left-[16%] right-[16%] h-px bg-gradient-to-r from-blue-500/50 via-purple-500/50 to-green-500/50" />

            {steps.map((step, idx) => (
              <motion.div
                key={step.title}
                variants={scaleIn}
                transition={{ duration: 0.5 }}
                className="relative text-center"
              >
                <div className="relative z-10 inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-slate-800 border border-white/10 mb-4">
                  <step.icon size={28} className="text-blue-400" />
                  <div className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-blue-600 text-xs font-bold flex items-center justify-center">
                    {idx + 1}
                  </div>
                </div>
                <h3 className="text-lg font-semibold mb-2">{step.title}</h3>
                <p className="text-gray-400 text-sm">{step.description}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="relative py-24 px-6">
        <div className="max-w-4xl mx-auto">
          <div className="relative p-12 rounded-3xl bg-gradient-to-br from-blue-600/20 via-purple-600/10 to-slate-900 border border-white/[0.06]">
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={staggerContainer}
              className="grid grid-cols-2 md:grid-cols-4 gap-8"
            >
              {stats.map((stat) => (
                <AnimatedCounter key={stat.label} {...stat} />
              ))}
            </motion.div>
          </div>
        </div>
      </section>

      {/* Final CTA Section */}
      <section className="relative py-32 px-6">
        <div className="max-w-3xl mx-auto text-center">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeUp}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-3xl md:text-5xl font-bold mb-6">
              Ready to make{' '}
              <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                smarter
              </span>{' '}
              decisions?
            </h2>
            <p className="text-gray-400 text-lg mb-10 max-w-xl mx-auto">
              Join and start simulating business scenarios with AI-powered insights in seconds.
            </p>
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.98 }}
            >
              <Link
                to="/login?signup=true"
                className="inline-flex items-center gap-2 px-10 py-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold text-lg rounded-xl transition-all shadow-lg shadow-blue-600/25 hover:shadow-blue-600/40"
              >
                Get Started Free
                <ArrowRight size={20} />
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/[0.06] py-8 px-6">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-md bg-blue-600 flex items-center justify-center">
              <Brain size={14} />
            </div>
            <span className="font-semibold text-sm">BizSimAI</span>
          </div>
          <p className="text-sm text-gray-500">AI-Driven Business Decision Simulation System</p>
        </div>
      </footer>
    </div>
  )
}

import { Routes, Route, NavLink } from 'react-router-dom'
import { BarChart3, MessageSquare, Upload, Home, GitBranch } from 'lucide-react'
import Dashboard from './pages/Home'
import Simulate from './pages/Simulate'
import ChatPage from './pages/Chat'
import UploadPage from './pages/Upload'
import Counterfactuals from './pages/Counterfactuals'

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b border-gray-200 px-6 py-3">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-primary-600" />
            <h1 className="text-lg font-bold text-gray-900">AI Business Simulator</h1>
          </div>
          <div className="flex gap-1">
            <NavItem to="/" icon={<Home size={18} />} label="Dashboard" />
            <NavItem to="/simulate" icon={<BarChart3 size={18} />} label="Simulate" />
            <NavItem to="/counterfactuals" icon={<GitBranch size={18} />} label="What-If" />
            <NavItem to="/chat" icon={<MessageSquare size={18} />} label="Chat" />
            <NavItem to="/upload" icon={<Upload size={18} />} label="Data" />
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-8">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/simulate" element={<Simulate />} />
          <Route path="/counterfactuals" element={<Counterfactuals />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/upload" element={<UploadPage />} />
        </Routes>
      </main>
    </div>
  )
}

function NavItem({ to, icon, label }: { to: string; icon: React.ReactNode; label: string }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
          isActive
            ? 'bg-primary-50 text-primary-700'
            : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
        }`
      }
    >
      {icon}
      {label}
    </NavLink>
  )
}

export default App

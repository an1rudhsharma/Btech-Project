import { useState, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Upload, FileText, Trash2, ArrowLeft, FileSpreadsheet, File, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { knowledgeApi } from '../api/client'
import toast from 'react-hot-toast'

export default function KnowledgeCenter() {
  const [dragActive, setDragActive] = useState(false)
  const [uploading, setUploading] = useState(false)
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: docs, isLoading } = useQuery({
    queryKey: ['knowledge-docs'],
    queryFn: () => knowledgeApi.list().then(r => r.data.documents),
  })

  const deleteMutation = useMutation({
    mutationFn: (docId: string) => knowledgeApi.delete(docId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['knowledge-docs'] })
      toast.success('Document deleted')
    },
  })

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault()
    setDragActive(false)
    const files = Array.from(e.dataTransfer.files)
    await uploadFiles(files)
  }, [])

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      await uploadFiles(Array.from(e.target.files))
      e.target.value = ''
    }
  }

  const uploadFiles = async (files: File[]) => {
    if (files.length === 0) return
    setUploading(true)
    let success = 0
    let failed = 0

    for (const file of files.slice(0, 20)) {
      try {
        await knowledgeApi.upload(file)
        success++
      } catch (e: any) {
        failed++
        toast.error(`Failed: ${file.name} — ${e.response?.data?.detail || e.message}`)
      }
    }

    setUploading(false)
    if (success > 0) {
      toast.success(`${success} file(s) processed successfully`)
      queryClient.invalidateQueries({ queryKey: ['knowledge-docs'] })
    }
  }

  const getFileIcon = (type: string) => {
    if (type === 'csv' || type === 'excel') return <FileSpreadsheet size={18} className="text-green-600" />
    if (type === 'pdf') return <FileText size={18} className="text-red-500" />
    return <File size={18} className="text-blue-500" />
  }

  const getStatusIcon = (status: string) => {
    if (status === 'ready') return <CheckCircle size={14} className="text-green-500" />
    if (status === 'error') return <AlertCircle size={14} className="text-red-500" />
    return <Loader2 size={14} className="text-blue-500 animate-spin" />
  }

  return (
    <div className="h-screen flex flex-col bg-[#f9fafb]">
      {/* Header */}
      <header className="border-b border-gray-200 bg-white px-6 py-4 flex items-center gap-4">
        <button onClick={() => navigate('/chat')} className="text-gray-500 hover:text-gray-700">
          <ArrowLeft size={20} />
        </button>
        <div>
          <h1 className="text-lg font-semibold text-gray-900">Knowledge Center</h1>
          <p className="text-xs text-gray-500">Upload documents to give the AI context about your business</p>
        </div>
      </header>

      <div className="flex-1 overflow-auto p-6 max-w-4xl mx-auto w-full">
        {/* Upload Zone */}
        <div
          onDragOver={e => { e.preventDefault(); setDragActive(true) }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-xl p-10 text-center transition-all cursor-pointer ${
            dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300 bg-white'
          }`}
          onClick={() => document.getElementById('kb-file-input')?.click()}
        >
          <input
            id="kb-file-input"
            type="file"
            multiple
            accept=".pdf,.txt,.docx,.csv,.xlsx,.xls,.md,.zip"
            className="hidden"
            onChange={handleFileSelect}
          />
          {uploading ? (
            <div className="flex flex-col items-center gap-3">
              <Loader2 size={40} className="text-blue-500 animate-spin" />
              <p className="text-sm text-gray-600">Processing files...</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <div className="w-14 h-14 rounded-full bg-blue-50 flex items-center justify-center">
                <Upload size={24} className="text-blue-500" />
              </div>
              <div>
                <p className="text-sm font-medium text-gray-700">
                  Drop files here or click to browse
                </p>
                <p className="text-xs text-gray-400 mt-1">
                  PDF, DOCX, TXT, CSV, Excel, ZIP — up to 50MB each
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Documents List */}
        <div className="mt-8">
          <h2 className="text-sm font-semibold text-gray-700 mb-3">
            Uploaded Documents ({docs?.length || 0})
          </h2>

          {isLoading ? (
            <div className="text-center py-8 text-gray-400 text-sm">Loading documents...</div>
          ) : !docs || docs.length === 0 ? (
            <div className="text-center py-12 text-gray-400">
              <FileText size={40} className="mx-auto mb-3 opacity-50" />
              <p className="text-sm">No documents uploaded yet</p>
              <p className="text-xs mt-1">Upload PDFs, reports, or data files to give the AI context</p>
            </div>
          ) : (
            <div className="space-y-2">
              {docs.map((doc: any) => (
                <div
                  key={doc.id}
                  className="flex items-center gap-3 bg-white border border-gray-100 rounded-lg px-4 py-3 hover:border-gray-200 transition-colors"
                >
                  {getFileIcon(doc.file_type)}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-800 truncate">{doc.filename}</p>
                    <p className="text-xs text-gray-400">
                      {doc.chunk_count} chunks
                      {doc.file_size_bytes ? ` • ${(doc.file_size_bytes / 1024).toFixed(0)} KB` : ''}
                      {doc.metadata?.queryable ? ' • Queryable' : ''}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    {getStatusIcon(doc.status)}
                    <button
                      onClick={(e) => { e.stopPropagation(); deleteMutation.mutate(doc.id) }}
                      className="p-1.5 text-gray-400 hover:text-red-500 transition-colors"
                      title="Delete document"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

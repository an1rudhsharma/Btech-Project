import { useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { useQuery } from '@tanstack/react-query'
import { uploadDataset, listDatasets, trainModel } from '../api/client'
import { Upload as UploadIcon, FileSpreadsheet, CheckCircle, Database } from 'lucide-react'
import toast from 'react-hot-toast'

export default function UploadPage() {
  const [uploadResult, setUploadResult] = useState<any>(null)
  const [uploading, setUploading] = useState(false)
  const [training, setTraining] = useState<string | null>(null)

  const { data: datasetsData, refetch } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => listDatasets().then(r => r.data),
  })

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { 'text/csv': ['.csv'], 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'] },
    maxFiles: 1,
    onDrop: async (files) => {
      if (files.length === 0) return
      setUploading(true)
      try {
        const res = await uploadDataset(files[0])
        setUploadResult(res.data)
        toast.success('Dataset uploaded successfully!')
        refetch()
      } catch (e: any) {
        toast.error(e.response?.data?.detail || 'Upload failed')
      } finally {
        setUploading(false)
      }
    },
  })

  const handleTrain = async (path: string, model: string) => {
    setTraining(`${model}-${path}`)
    try {
      await trainModel(path, model)
      toast.success(`${model} model trained successfully!`)
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Training failed')
    } finally {
      setTraining(null)
    }
  }

  const datasets = datasetsData?.datasets || []

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Data Management</h2>
        <p className="text-gray-600 mt-1">Upload datasets and train models</p>
      </div>

      {/* Upload Zone */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors ${
          isDragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
        }`}
      >
        <input {...getInputProps()} />
        <UploadIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        {uploading ? (
          <p className="text-gray-600">Uploading...</p>
        ) : isDragActive ? (
          <p className="text-primary-600 font-medium">Drop the file here</p>
        ) : (
          <>
            <p className="text-gray-600">Drag & drop a CSV or Excel file here</p>
            <p className="text-sm text-gray-400 mt-2">or click to browse</p>
          </>
        )}
      </div>

      {/* Upload Result */}
      {uploadResult && (
        <div className="bg-white rounded-xl p-6 border border-gray-200">
          <div className="flex items-center gap-2 mb-4">
            <CheckCircle className="w-5 h-5 text-green-500" />
            <h3 className="font-semibold text-gray-900">Uploaded: {uploadResult.filename}</h3>
          </div>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="text-sm"><span className="text-gray-500">Rows:</span> <strong>{uploadResult.rows}</strong></div>
            <div className="text-sm"><span className="text-gray-500">Columns:</span> <strong>{uploadResult.columns?.length}</strong></div>
          </div>
          <div className="mb-4">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Detected Column Mapping:</h4>
            <div className="flex flex-wrap gap-2">
              {Object.entries(uploadResult.detected_mapping || {}).map(([role, col]) =>
                col ? (
                  <span key={role} className="text-xs bg-green-50 text-green-700 border border-green-200 rounded-full px-3 py-1">
                    {role} → {String(col)}
                  </span>
                ) : null
              )}
            </div>
          </div>
          <div className="flex gap-2">
            {['churn', 'marketing', 'pricing', 'sentiment'].map(model => (
              <button
                key={model}
                onClick={() => handleTrain(uploadResult.path, model)}
                disabled={training === `${model}-${uploadResult.path}`}
                className="px-3 py-2 text-sm bg-primary-50 text-primary-700 rounded-lg hover:bg-primary-100 disabled:opacity-50 capitalize"
              >
                Train {model}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Existing Datasets */}
      <div className="bg-white rounded-xl p-6 border border-gray-200">
        <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Database size={18} /> Available Datasets
        </h3>
        {datasets.length > 0 ? (
          <div className="space-y-3">
            {datasets.map((ds: any, i: number) => (
              <div key={i} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-3">
                  <FileSpreadsheet className="w-5 h-5 text-gray-400" />
                  <div>
                    <p className="text-sm font-medium text-gray-900">{ds.name}</p>
                    <p className="text-xs text-gray-500">{ds.rows} rows · {ds.columns?.length} columns · {ds.type}</p>
                  </div>
                </div>
                <div className="flex gap-2">
                  {['churn', 'marketing', 'pricing', 'sentiment'].map(model => (
                    <button
                      key={model}
                      onClick={() => handleTrain(ds.path, model)}
                      disabled={training === `${model}-${ds.path}`}
                      className="px-2 py-1 text-xs bg-white border border-gray-200 text-gray-600 rounded hover:bg-gray-100 capitalize"
                    >
                      {model}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-400">No datasets found. Upload one or train with sample data from the Dashboard.</p>
        )}
      </div>
    </div>
  )
}

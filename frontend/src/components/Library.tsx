import { useEffect, useState } from 'react'
import { api, uploadPdf, type DocMeta } from '../api'

export function Library({ onOpen }: { onOpen: (id: string) => void }) {
  const [docs, setDocs] = useState<DocMeta[] | null>(null)
  const [progress, setProgress] = useState<{ name: string; fraction: number } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)

  useEffect(() => {
    api.listDocs().then(setDocs, (e) => setError(`Backend unreachable: ${e.message}`))
  }, [])

  const upload = async (files: FileList | null) => {
    const pdfs = [...(files ?? [])].filter((f) => f.name.toLowerCase().endsWith('.pdf'))
    if (!pdfs.length) return setError('Choose a PDF file.')
    setError(null)
    let last: DocMeta | null = null
    for (const f of pdfs) {
      setProgress({ name: f.name, fraction: 0 })
      try {
        last = await uploadPdf(f, (fraction) => setProgress({ name: f.name, fraction }))
      } catch (e) {
        setError(`${f.name}: ${(e as Error).message}`)
      }
    }
    setProgress(null)
    if (pdfs.length === 1 && last) onOpen(last.id)
    else api.listDocs().then(setDocs)
  }

  return (
    <main className="library">
      <header>
        <h1>docparse</h1>
        <p className="muted">Open a PDF, select what you need, and save it as a reusable template.</p>
      </header>

      <label
        className={`dropzone ${dragging ? 'over' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          upload(e.dataTransfer.files)
        }}
      >
        <input
          type="file"
          accept="application/pdf"
          multiple
          className="visually-hidden"
          onChange={(e) => upload(e.target.files)}
        />
        {progress ? (
          <>
            <b>
              {progress.fraction < 1 ? 'Uploading' : 'Indexing'} {progress.name}
            </b>
            <div className="bar">
              <div style={{ width: `${progress.fraction * 100}%` }} />
            </div>
          </>
        ) : (
          <>
            <b>Drop PDFs here</b>
            <span className="button secondary">Choose PDF</span>
            <span className="muted small">Up to 200 MB each</span>
          </>
        )}
      </label>

      {error && <p className="error">{error}</p>}

      <section>
        <h2>Documents</h2>
        {docs === null && !error && <p className="muted">Loading…</p>}
        {docs?.length === 0 && <p className="muted">Nothing uploaded yet.</p>}
        <ul className="doc-list">
          {docs?.map((d) => (
            <li key={d.id}>
              <button onClick={() => onOpen(d.id)}>
                <span className="doc-name">{d.file_name}</span>
                <span className="muted meta">
                  {d.page_count} pages · {(d.size_bytes / 1e6).toFixed(1)} MB ·{' '}
                  {new Date(d.uploaded_at).toLocaleDateString()}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </section>
    </main>
  )
}

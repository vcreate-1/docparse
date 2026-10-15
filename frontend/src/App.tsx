import { Suspense, lazy, useEffect, useState } from 'react'
import { api } from './api'
import { Library } from './components/Library'

// PDF.js is ~600 KB; load it only when a document is opened.
const Workspace = lazy(() =>
  import('./components/Workspace').then((m) => ({ default: m.Workspace })),
)

// Minimal hash routing (#/doc/<sha256>) so a reload keeps the open document.
const docFromHash = () => location.hash.match(/^#\/doc\/([0-9a-f]{64})$/)?.[1] ?? null

export default function App() {
  const [docId, setDocId] = useState(docFromHash)

  useEffect(() => {
    const onHash = () => setDocId(docFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  // Fetch metadata in parallel with the lazy viewer chunk instead of after it.
  if (docId) void api.getDoc(docId).catch(() => {})

  return docId ? (
    <Suspense fallback={<div className="center muted">Loading viewer…</div>}>
      <Workspace key={docId} docId={docId} onBack={() => (location.hash = '')} />
    </Suspense>
  ) : (
    <Library onOpen={(id) => (location.hash = `/doc/${id}`)} />
  )
}

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { api, type Detector, type DocMeta, type RegionKind, type Target, type Template } from '../api'
import { SidePanel, type Tab } from './SidePanel'
import { KIND_LABELS, Viewer, type DetectorStats, type Selection } from './Viewer'

const TOGGLEABLE: RegionKind[] = ['table', 'chart_vector', 'image', 'heading', 'text', 'scanned_page']
const MIN_SCALE = 0.25
const MAX_SCALE = 4

export function Workspace({ docId, onBack }: { docId: string; onBack: () => void }) {
  const [doc, setDoc] = useState<DocMeta | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [scale, setScale] = useState<number | null>(null)
  const [page, setPage] = useState(1)
  const [pageInput, setPageInput] = useState('1')
  const [jump, setJump] = useState<{ page: number; nonce: number } | null>(null)
  const [visibleKinds, setVisibleKinds] = useState(
    () => new Set<RegionKind>(TOGGLEABLE.filter((k) => k !== 'text')),
  )
  const [selection, setSelection] = useState<Selection | null>(null)
  const [drawMode, setDrawMode] = useState(false)
  const [tab, setTab] = useState<Tab>('find')
  const [targets, setTargets] = useState<Target[]>([])
  const [templateName, setTemplateName] = useState('')
  const [template, setTemplate] = useState<Template | null>(null)
  const [savedSnapshot, setSavedSnapshot] = useState('[]')
  const [detectors, setDetectors] = useState<Detector[]>([])
  const [detector, setDetector] = useState('auto')
  const [compareDetector, setCompareDetector] = useState<string | null>(null)
  const [detectorStats, setDetectorStats] = useState<DetectorStats[]>([])
  const searchRef = useRef<HTMLInputElement>(null)
  const viewerBox = useRef<HTMLDivElement>(null)

  const dirty = JSON.stringify(targets) !== savedSnapshot
  const available = detectors.filter((d) => d.available)

  useEffect(() => {
    api.getDoc(docId).then(setDoc, (e) => setError(e.message))
    api.detectors().then(setDetectors, () => {})
  }, [docId])

  useEffect(() => {
    if (!dirty) return
    const warn = (e: BeforeUnloadEvent) => e.preventDefault()
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [dirty])

  const back = () => {
    if (!dirty || confirm('Discard unsaved template changes?')) onBack()
  }

  // Snapshot what was sent, not the server echo: the server fills in defaults
  // (e.g. page_keywords: []) that would make the template look permanently dirty.
  const onSaved = (t: Template, sent: Target[]) => {
    setTemplate(t)
    setSavedSnapshot(JSON.stringify(sent))
  }

  const fitWidth = useCallback(() => {
    if (!doc || !viewerBox.current) return
    const widest = Math.max(...doc.page_sizes.map(([w]) => w))
    setScale(Math.max(MIN_SCALE, Math.min(MAX_SCALE, (viewerBox.current.clientWidth - 48) / widest)))
  }, [doc])

  useLayoutEffect(() => {
    if (doc && scale === null) fitWidth()
  }, [doc, scale, fitWidth])

  const goTo = useCallback((p: number) => {
    setJump({ page: p, nonce: Math.random() })
  }, [])

  const onPageChange = useCallback((p: number) => {
    setPage(p)
    setPageInput(String(p))
  }, [])

  const onSelect = useCallback((s: Selection) => {
    setSelection(s)
    setDrawMode(false)
    setTab('selection')
  }, [])

  const zoom = useCallback(
    (factor: number) =>
      setScale((s) => Math.max(MIN_SCALE, Math.min(MAX_SCALE, (s ?? 1) * factor))),
    [],
  )

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = (e.target as HTMLElement).closest('input, textarea, select')
      if (e.key === 'Escape') {
        if (typing) (e.target as HTMLElement).blur()
        else if (drawMode) setDrawMode(false)
        else setSelection(null)
        return
      }
      if (typing || e.metaKey || e.ctrlKey) return
      if (e.key === 'd') setDrawMode((d) => !d)
      else if (e.key === '/') {
        e.preventDefault()
        setTab('find')
        setTimeout(() => searchRef.current?.focus())
      } else if (e.key === '+' || e.key === '=') zoom(1.2)
      else if (e.key === '-') zoom(1 / 1.2)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [zoom, drawMode])

  if (error) return <div className="center error">{error}</div>

  const compareStats =
    compareDetector &&
    detectorStats
      .map((s) => `${s.loaded ? Math.round(s.avgMs) : '–'} ms/page`)
      .join(' vs ')

  return (
    <div className="workspace">
      <header className="toolbar">
        <div className="group">
          <button className="ghost" onClick={back} aria-label="Back to documents">
            ← <span className="wide-only">Documents</span>
          </button>
          <span className="doc-title" title={doc?.file_name}>
            {doc?.file_name ?? 'Loading…'}
          </span>
        </div>
        {doc && (
          <>
            <div className="group">
              <form
                className="page-nav"
                onSubmit={(e) => {
                  e.preventDefault()
                  const n = Number(pageInput)
                  if (n >= 1 && n <= doc.page_count) goTo(n)
                  else setPageInput(String(page))
                }}
              >
                <input
                  value={pageInput}
                  inputMode="numeric"
                  onChange={(e) => setPageInput(e.target.value.replace(/\D/g, ''))}
                  aria-label="Page number"
                />
                <span className="muted">/ {doc.page_count}</span>
              </form>
              <div className="zoom">
                <button className="ghost" onClick={() => zoom(1 / 1.2)} aria-label="Zoom out" title="Zoom out (−)">
                  −
                </button>
                <span aria-live="polite">{Math.round((scale ?? 1) * 100)}%</span>
                <button className="ghost" onClick={() => zoom(1.2)} aria-label="Zoom in" title="Zoom in (+)">
                  +
                </button>
                <button className="ghost" onClick={fitWidth}>
                  Fit
                </button>
              </div>
            </div>
            <div className="group">
              <button
                className={drawMode ? 'primary' : 'secondary'}
                aria-pressed={drawMode}
                onClick={() => setDrawMode((d) => !d)}
              >
                {drawMode ? 'Drawing · Esc to stop' : 'Draw region'} {!drawMode && <kbd>D</kbd>}
              </button>
            </div>
            <div className="group">
              <Layers
                visibleKinds={visibleKinds}
                setVisibleKinds={setVisibleKinds}
                available={available}
                detector={detector}
                setDetector={setDetector}
                compareDetector={compareDetector}
                setCompareDetector={setCompareDetector}
                compareStats={compareStats || null}
              />
            </div>
          </>
        )}
      </header>
      <div className="main">
        <div className="viewer-box" ref={viewerBox}>
          {doc && scale !== null && (
            <Viewer
              doc={doc}
              scale={scale}
              visibleKinds={visibleKinds}
              selection={selection}
              targets={targets}
              drawMode={drawMode}
              jump={jump}
              onSelect={onSelect}
              onPageChange={onPageChange}
              detector={detector}
              compareDetector={compareDetector}
              onRegionsProgress={setDetectorStats}
            />
          )}
        </div>
        {doc && (
          <SidePanel
            doc={doc}
            tab={tab}
            setTab={setTab}
            selection={selection}
            targets={targets}
            setTargets={setTargets}
            templateName={templateName}
            setTemplateName={setTemplateName}
            template={template}
            onSaved={onSaved}
            dirty={dirty}
            searchRef={searchRef}
            onJump={goTo}
          />
        )}
      </div>
    </div>
  )
}

/** Region-type toggles and detector choice: occasional settings, kept out of the toolbar. */
function Layers(p: {
  visibleKinds: Set<RegionKind>
  setVisibleKinds: (s: Set<RegionKind>) => void
  available: Detector[]
  detector: string
  setDetector: (d: string) => void
  compareDetector: string | null
  setCompareDetector: (d: string | null) => void
  compareStats: string | null
}) {
  const ref = useRef<HTMLDetailsElement>(null)
  useEffect(() => {
    const close = (e: PointerEvent) => {
      if (ref.current?.open && !ref.current.contains(e.target as Node)) ref.current.open = false
    }
    document.addEventListener('pointerdown', close)
    return () => document.removeEventListener('pointerdown', close)
  }, [])

  return (
    <details className="layers" ref={ref}>
      <summary>
        Layers
        <span className="legend" aria-hidden="true">
          {TOGGLEABLE.filter((k) => p.visibleKinds.has(k)).map((k) => (
            <i key={k} className={`kind-${k}`} />
          ))}
        </span>
        {p.compareDetector && <span className="muted">· comparing</span>}
      </summary>
      <div className="popover">
        <p className="popover-title">Show</p>
        <div className="kinds" role="group" aria-label="Show region types">
          {TOGGLEABLE.map((k) => (
            <button
              key={k}
              className={`chip kind-${k} ${p.visibleKinds.has(k) ? 'on' : 'off'}`}
              aria-pressed={p.visibleKinds.has(k)}
              onClick={() => {
                const next = new Set(p.visibleKinds)
                if (next.has(k)) next.delete(k)
                else next.add(k)
                p.setVisibleKinds(next)
              }}
            >
              {KIND_LABELS[k]}
            </button>
          ))}
        </div>
        {/* Detector choice only exists when there's more than one to choose. */}
        {p.available.length > 1 && (
          <div className="detector">
            <label className="field">
              Detector
              <select value={p.detector} onChange={(e) => p.setDetector(e.target.value)}>
                {p.available.map((d) => (
                  <option key={d.id} value={d.id} title={d.description}>
                    {d.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              Compare with (dashed)
              <select
                value={p.compareDetector ?? ''}
                onChange={(e) => p.setCompareDetector(e.target.value || null)}
              >
                <option value="">Off</option>
                {p.available
                  .filter((d) => d.id !== p.detector)
                  .map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.label}
                    </option>
                  ))}
              </select>
            </label>
            {p.compareStats && <p className="muted small compare-stats">{p.compareStats}</p>}
          </div>
        )}
      </div>
    </details>
  )
}

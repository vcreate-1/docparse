import { useEffect, useRef, useState } from 'react'
import {
  api,
  type Confidence,
  type DocMeta,
  type ExtractResult,
  type FieldMatch,
  type FieldSpec,
  type FieldType,
  type RegionKind,
  type SearchHit,
  type Target,
  type Template,
} from '../api'
import { KIND_LABELS, type Selection } from './Viewer'

export type Tab = 'find' | 'selection' | 'template'

interface Props {
  doc: DocMeta
  tab: Tab
  setTab: (t: Tab) => void
  selection: Selection | null
  targets: Target[]
  setTargets: (t: Target[]) => void
  templateName: string
  setTemplateName: (n: string) => void
  template: Template | null
  onSaved: (t: Template, sent: Target[]) => void
  dirty: boolean
  searchRef: React.RefObject<HTMLInputElement | null>
  onJump: (page: number) => void
}

export function SidePanel(p: Props) {
  const tabs: [Tab, string][] = [
    ['find', 'Find'],
    ['selection', 'Selection'],
    ['template', `Template (${p.targets.length})${p.dirty ? ' •' : ''}`],
  ]
  return (
    <aside className="side">
      <nav className="tabs" role="tablist">
        {tabs.map(([t, label]) => (
          <button
            key={t}
            role="tab"
            aria-selected={p.tab === t}
            className={p.tab === t ? 'active' : ''}
            onClick={() => p.setTab(t)}
          >
            {label}
          </button>
        ))}
      </nav>
      <div className="side-body" role="tabpanel">
        {p.tab === 'find' && <FindTab {...p} />}
        {p.tab === 'selection' && <SelectionTab {...p} />}
        {p.tab === 'template' && <TemplateTab {...p} />}
      </div>
    </aside>
  )
}

function FindTab({ doc, searchRef, onJump }: Props) {
  const [q, setQ] = useState('')
  const [hits, setHits] = useState<SearchHit[] | null>(null)

  useEffect(() => {
    if (!q.trim()) {
      setHits(null)
      return
    }
    let stale = false
    const t = setTimeout(async () => {
      const res = await api.search(doc.id, q).catch(() => [])
      if (!stale) setHits(res)
    }, 120)
    return () => {
      stale = true
      clearTimeout(t)
    }
  }, [q, doc.id])

  const terms = q.toLowerCase().split(/\s+/).filter((t) => t.length > 1)
  const missingText = doc.page_count - doc.text_pages

  return (
    <div className="find">
      <label className="field">
        Search pages
        <input
          ref={searchRef}
          type="search"
          placeholder="e.g. segment revenue"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          autoFocus
        />
      </label>
      {missingText > 0 && (
        <p className="note warn-note">
          {missingText} of {doc.page_count} pages have no text layer and won’t appear in search.
        </p>
      )}
      {hits && (
        <p className="muted small">
          {hits.length ? `${hits.length} page${hits.length === 1 ? '' : 's'}` : 'No matching pages'}
        </p>
      )}
      <ul className="hits">
        {hits?.map((h) => (
          <li key={h.page}>
            <button onClick={() => onJump(h.page)}>
              <span className="hit-page">p. {h.page}</span>
              <span className="hit-snippet">{highlight(h.snippet, terms)}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}

function highlight(text: string, terms: string[]) {
  if (!terms.length) return text
  const re = new RegExp(`(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi')
  return text.split(re).map((part, i) => (i % 2 ? <mark key={i}>{part}</mark> : part))
}

const CUSTOM_KINDS: RegionKind[] = ['text', 'table', 'chart_vector', 'image']

// Glyph + sentence: confidence must read without relying on colour.
const CONFIDENCE: Record<Confidence, { glyph: string; label: string }> = {
  exact: { glyph: '✓', label: 'Exact' },
  near_exact: { glyph: '≈', label: 'Near exact' },
  approximate: { glyph: '⚠', label: 'Approximate' },
}

function confidenceReason(r: ExtractResult): string {
  const ocr = r.method.startsWith('ocr')
  const engine = r.method.includes('cortex') ? 'Snowflake Cortex' : 'PaddleOCR'
  if (r.confidence === 'exact') {
    return r.type === 'chart'
      ? 'values read from the chart’s data labels'
      : 'values copied from the PDF text layer'
  }
  if (r.confidence === 'near_exact') {
    return ocr
      ? `read from the picture by ${engine} (${Math.round((r.ocr_confidence ?? 0) * 100)}% confidence); spot-check a value`
      : 'computed from chart geometry; spot-check a value'
  }
  if (ocr && r.issues?.length) return `check against the page: ${r.issues.join('; ')}`
  return 'estimated; check against the page before use'
}

function SelectionTab({ doc, selection, targets, setTargets }: Props) {
  const [result, setResult] = useState<ExtractResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [treatAs, setTreatAs] = useState<RegionKind>('text')
  const [name, setName] = useState('')
  const [instruction, setInstruction] = useState('')
  const [fieldSpecs, setFieldSpecs] = useState<FieldSpec[]>([])
  const [rowsMode, setRowsMode] = useState(false)
  const [fieldResult, setFieldResult] = useState<ExtractResult | null>(null)
  const [extracting, setExtracting] = useState(false)
  const [llmReady, setLlmReady] = useState<boolean | null>(null)
  const nameTouched = useRef(false)

  const kind = selection?.kind === 'custom' ? treatAs : selection?.kind
  const isText = kind === 'text' || kind === 'heading'
  const usableFields = fieldSpecs.filter((f) => /^[A-Za-z_][A-Za-z0-9_]*$/.test(f.name))

  useEffect(() => {
    api.llm().then((l) => setLlmReady(l.configured), () => setLlmReady(false))
  }, [])

  const extractFields = async () => {
    if (!selection || !kind) return
    setExtracting(true)
    setError(null)
    try {
      setFieldResult(
        await api.extract(doc.id, selection.page, selection.bbox, kind, {
          fields: usableFields,
          instruction: instruction || undefined,
          rows: rowsMode,
        }),
      )
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setExtracting(false)
    }
  }
  const alreadyAdded =
    !!selection &&
    targets.some(
      (t) => t.page === selection.page && t.bbox.every((v, i) => Math.abs(v - selection.bbox[i]) < 0.002),
    )

  useEffect(() => {
    if (!selection || !kind) return
    let stale = false
    setLoading(true) // the previous result stays on screen, dimmed, until this one lands
    setError(null)
    api
      .extract(doc.id, selection.page, selection.bbox, kind)
      .then((r) => {
        if (stale) return
        setResult(r)
        if (!nameTouched.current) setName(slug(r.title || r.anchor_heading || selection.label || kind))
      })
      .catch((e) => !stale && setError(e.message))
      .finally(() => !stale && setLoading(false))
    return () => {
      stale = true
    }
  }, [doc.id, selection, kind])

  useEffect(() => {
    nameTouched.current = false
    setInstruction('')
    setFieldResult(null)
  }, [selection])

  if (!selection) {
    return (
      <p className="empty">
        Click a box on the page, or press <kbd>D</kbd> and drag around anything. Boxes can also be
        reached with <kbd>Tab</kbd>.
      </p>
    )
  }

  const add = () => {
    if (!result || !kind || alreadyAdded) return
    setTargets([
      ...targets,
      {
        name: name || slug(kind),
        kind,
        method: result.method,
        page: selection.page,
        bbox: selection.bbox,
        instruction: instruction || undefined,
        anchor_heading: result.title || result.anchor_heading,
        header: result.columns ?? [],
        ...(isText && usableFields.length ? { fields: usableFields, rows: rowsMode } : {}),
      },
    ])
  }

  const anchor = result?.title || result?.anchor_heading
  const conf = result && CONFIDENCE[result.confidence]

  return (
    <div
      className={`selection kind-${selection.kind}`}
      onKeyDown={(e) => {
        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) add()
      }}
    >
      <div className="sel-head">
        <b>{KIND_LABELS[kind ?? selection.kind]}</b>
        <span className="muted">
          p. {selection.page}
          {result && ` · ${result.method}`}
        </span>
      </div>

      {result && conf && (
        <p className={`confidence ${result.confidence}`}>
          <span aria-hidden="true">{conf.glyph}</span> <b>{conf.label}:</b> {confidenceReason(result)}
        </p>
      )}

      {selection.kind === 'custom' && (
        <label className="field">
          Read this region as
          <select value={treatAs} onChange={(e) => setTreatAs(e.target.value as RegionKind)}>
            {CUSTOM_KINDS.map((k) => (
              <option key={k} value={k}>
                {KIND_LABELS[k]}
              </option>
            ))}
          </select>
        </label>
      )}

      {loading && !result && <p className="muted loading">Extracting…</p>}
      {error && <p className="error">{error}</p>}

      {result && (
        <div className={`result ${loading ? 'stale' : ''}`} key={`${selection.page}-${selection.bbox.join()}`}>
          <dl className="facts">
            <dt>Anchor</dt>
            <dd>
              {anchor || (
                <span className="warn">None found. Future documents will be matched by page position only.</span>
              )}
            </dd>
          </dl>
          {result.note && <p className="note">{result.note}</p>}
          <Preview result={result} />
        </div>
      )}

      {isText && result && (
        <section className="data-fields">
          <h4>Data to extract</h4>
          <p className="muted small">
            The language model reads this text and returns these fields; each value is checked
            word-for-word against the text.
          </p>
          {fieldSpecs.map((f, i) => (
            <div className="field-row" key={i}>
              <input
                aria-label="Field name"
                placeholder="e.g. gdp_growth"
                value={f.name}
                onChange={(e) =>
                  setFieldSpecs(fieldSpecs.map((x, j) => (j === i ? { ...x, name: e.target.value.replace(/\W/g, '_') } : x)))
                }
              />
              <select
                aria-label="Field type"
                value={f.type}
                onChange={(e) =>
                  setFieldSpecs(fieldSpecs.map((x, j) => (j === i ? { ...x, type: e.target.value as FieldType } : x)))
                }
              >
                {['text', 'number', 'percent', 'date', 'boolean', 'list'].map((t) => (
                  <option key={t}>{t}</option>
                ))}
              </select>
              <button className="remove-inline" aria-label={`Remove ${f.name || 'field'}`}
                onClick={() => setFieldSpecs(fieldSpecs.filter((_, j) => j !== i))}>
                ✕
              </button>
            </div>
          ))}
          <div className="row">
            <button className="secondary" onClick={() => setFieldSpecs([...fieldSpecs, { name: '', type: 'number' }])}>
              + Add field
            </button>
            <label className="inline">
              <input type="checkbox" checked={rowsMode} onChange={(e) => setRowsMode(e.target.checked)} />
              One row per item
            </label>
          </div>
          {llmReady === false && (
            <p className="note warn-note">
              No language model is configured (RAG facility, OpenAI or LM Studio), so fields can’t be
              extracted yet. The text above is exact.
            </p>
          )}
          <button
            className="primary block"
            disabled={!usableFields.length || extracting || llmReady === false}
            onClick={extractFields}
          >
            {extracting ? 'Extracting… (a local model can take ~20 s)' : 'Extract fields'}
          </button>
          {fieldResult && (
            <div className="result">
              <p className={`confidence ${fieldResult.confidence}`}>
                <span aria-hidden="true">{CONFIDENCE[fieldResult.confidence].glyph}</span>{' '}
                <b>{CONFIDENCE[fieldResult.confidence].label}:</b> {fieldResult.note}
              </p>
              <Preview result={{ ...fieldResult, type: 'table' }} />
              <p className="muted small">{fieldResult.method.replace('llm:', 'Model: ')}{fieldResult.cached ? ' · cached' : ''}</p>
            </div>
          )}
        </section>
      )}

      <div className="add">
        <label className="field">
          Field name
          <input
            value={name}
            onChange={(e) => {
              nameTouched.current = true
              setName(e.target.value)
            }}
          />
        </label>
        <label className="field">
          <span>
            Instruction <span className="muted">(optional)</span>
          </span>
          <textarea
            className={instruction ? 'has-value' : ''}
            rows={1}
            placeholder="e.g. Only Tesla and Toyota rows"
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
          />
        </label>
        <button className="primary block" disabled={!result || loading || alreadyAdded} onClick={add}>
          {alreadyAdded ? 'Added ✓' : 'Add to template'} {!alreadyAdded && <kbd>⌘↵</kbd>}
        </button>
      </div>
    </div>
  )
}

function Preview({ result }: { result: ExtractResult }) {
  if (result.type === 'table') {
    if (!result.data?.length) return <p className="muted">No rows.</p>
    return (
      <div className="table-wrap">
        <p className="muted small">
          {result.data.length} rows × {result.columns?.length} columns
        </p>
        <table>
          <thead>
            <tr>
              {result.columns?.map((c, i) => <th key={i}>{c}</th>)}
            </tr>
          </thead>
          <tbody>
            {result.data.slice(0, 200).map((row, i) => (
              <tr key={i}>
                {row.map((c, j) => <td key={j}>{c}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }
  if (result.type === 'text') {
    return <pre className="text-preview">{result.text || '(no text in this region)'}</pre>
  }
  if (result.data?.length) {
    // Chart values read as label / value rows (OCR of printed data labels).
    return <Preview result={{ ...result, type: 'table' }} />
  }
  return (
    <div className="chart-preview">
      {!!result.numeric_labels?.length && (
        <>
          <h4>Values</h4>
          <p className="values">{result.numeric_labels.join(' · ')}</p>
        </>
      )}
      {!!result.labels?.length && (
        <>
          <h4>All text in region</h4>
          <p className="muted">{result.labels.join(' ')}</p>
        </>
      )}
    </div>
  )
}

function TemplateTab({ doc, targets, setTargets, templateName, setTemplateName, template, onSaved, dirty, onJump }: Props) {
  const [saving, setSaving] = useState(false)
  const [status, setStatus] = useState<string | null>(null)

  const save = async () => {
    setSaving(true)
    setStatus(null)
    try {
      const saved = await api.saveTemplate(
        { name: templateName.trim(), source_doc_id: doc.id, targets },
        template?.id,
      )
      onSaved(saved, targets)
      setStatus(`Saved as version ${saved.version}`)
    } catch (e) {
      setStatus(`Save failed: ${(e as Error).message}`)
    } finally {
      setSaving(false)
    }
  }

  const copyJson = async () => {
    await navigator.clipboard.writeText(
      JSON.stringify({ name: templateName, source_doc_id: doc.id, targets }, null, 2),
    )
    setStatus('JSON copied')
  }

  return (
    <div className="template">
      <label className="field">
        Template name
        <input
          value={templateName}
          placeholder="e.g. KBB EV quarterly sales"
          onChange={(e) => setTemplateName(e.target.value)}
        />
      </label>
      {targets.length === 0 ? (
        <p className="empty">No fields yet. Select a box on the page and choose “Add to template”.</p>
      ) : (
        <ol className="targets">
          {targets.map((t, i) => (
            <li key={i}>
              <button className="target-main" onClick={() => onJump(t.page)}>
                <b>{t.name}</b>
                <span className="muted">
                  {KIND_LABELS[t.kind]} · p. {t.page}
                </span>
                <span className="muted">
                  Anchor: {t.anchor_heading ?? <span className="warn">none</span>}
                </span>
                {t.instruction && <span className="instr">“{t.instruction}”</span>}
              </button>
              <button
                className="remove"
                aria-label={`Remove ${t.name}`}
                onClick={() => setTargets(targets.filter((_, j) => j !== i))}
              >
                ✕
              </button>
            </li>
          ))}
        </ol>
      )}
      <button
        className="primary block"
        disabled={!targets.length || !templateName.trim() || saving || (!!template && !dirty)}
        onClick={save}
      >
        {template ? (dirty ? `Save version ${template.version + 1}` : `Saved · version ${template.version}`) : 'Save template'}
      </button>
      {status && (
        <p className="muted small" role="status">
          {status}
        </p>
      )}
      {targets.length > 0 && (
        <button className="link" onClick={copyJson}>
          Copy as JSON
        </button>
      )}
      {template && !dirty && (
        <DryRun template={template} doc={doc} targets={targets} onSaved={onSaved} />
      )}
    </div>
  )
}

const STATUS: Record<FieldMatch['status'], { glyph: string; label: string }> = {
  matched: { glyph: '✓', label: 'Matched' },
  ambiguous: { glyph: '⚠', label: 'Needs review' },
  not_found: { glyph: '✕', label: 'Not found' },
}

/** Run the saved template against another document: does every field anchor? */
function DryRun({
  template,
  doc,
  targets,
  onSaved,
}: {
  template: Template
  doc: DocMeta
  targets: Target[]
  onSaved: (t: Template, sent: Target[]) => void
}) {
  const [docs, setDocs] = useState<DocMeta[]>([])
  const [other, setOther] = useState('')
  const [running, setRunning] = useState(false)
  const [results, setResults] = useState<FieldMatch[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.listDocs().then((all) => setDocs(all.filter((d) => d.id !== doc.id)), () => {})
  }, [doc.id])

  const run = async () => {
    setRunning(true)
    setError(null)
    setResults(null)
    try {
      setResults((await api.locate(template.id, other)).fields)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setRunning(false)
    }
  }

  const confirm = async (i: number, m: FieldMatch) => {
    if (!m.best) return
    const saved = await api.confirm(template.id, i, other, m.best.page, m.best.bbox)
    onSaved(saved, targets)
    setResults((rs) => rs && rs.map((r, j) => (j === i ? { ...r, status: 'matched' } : r)))
  }

  return (
    <section className="dry-run">
      <h3>Test on another document</h3>
      <p className="muted small">
        Finds each field by what it is (heading, columns, row labels), not where it was.
      </p>
      <div className="row">
        <select value={other} onChange={(e) => setOther(e.target.value)} aria-label="Document to test">
          <option value="">Choose a document…</option>
          {docs.map((d) => (
            <option key={d.id} value={d.id}>
              {d.file_name}
            </option>
          ))}
        </select>
        <button className="secondary" disabled={!other || running} onClick={run}>
          {running ? 'Running…' : 'Run'}
        </button>
      </div>
      {error && <p className="error">{error}</p>}
      {results && (
        <ol className="matches">
          {results.map((m, i) => (
            <li key={i} className={`match ${m.status}`}>
              <div className="match-head">
                <b>{m.name}</b>
                <span className={`status ${m.status}`}>
                  <span aria-hidden="true">{STATUS[m.status].glyph}</span> {STATUS[m.status].label}
                </span>
              </div>
              {m.best && (
                <p className="muted small">
                  p. {m.best.page} · score {Math.round(m.best.score * 100)}% · lead over next{' '}
                  {Math.round(m.margin * 100)} pts · matched on “{m.best.anchor || 'position'}”
                </p>
              )}
              {m.values && <Preview result={m.values} />}
              {m.status === 'ambiguous' && m.best && (
                <button className="secondary" onClick={() => confirm(i, m)}>
                  Confirm this match
                </button>
              )}
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}

function slug(s: string) {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_|_$/g, '')
    .slice(0, 60)
}

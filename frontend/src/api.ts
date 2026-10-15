export type BBox = [number, number, number, number] // normalised 0..1, top-left origin

export type RegionKind =
  | 'table'
  | 'chart_vector'
  | 'image'
  | 'heading'
  | 'text'
  | 'scanned_page'
  | 'custom'

export interface DocMeta {
  id: string
  file_name: string
  title: string
  page_count: number
  page_sizes: [number, number][]
  size_bytes: number
  text_pages: number
  uploaded_at: string
  deduplicated?: boolean
}

export interface Detector {
  id: string
  label: string
  description: string
  available: boolean
}

export interface Region {
  id: string
  kind: RegionKind
  bbox: BBox
  method: string
  label: string
  title?: string | null
  header?: string[]
  model_label?: string
  score?: number
}

export interface PageRegions {
  page: number
  scanned: boolean
  regions: Region[]
  ms: number
}

export interface SearchHit {
  page: number
  score: number
  snippet: string
}

export type Confidence = 'exact' | 'near_exact' | 'approximate'

export interface ExtractResult {
  type: 'table' | 'text' | 'chart' | 'fields'
  method: string
  confidence: Confidence
  anchor_heading: string | null
  note?: string
  ocr_confidence?: number
  issues?: string[]
  cached?: boolean
  // table
  title?: string | null
  columns?: string[]
  data?: string[][]
  // text
  text?: string
  // chart
  labels?: string[]
  numeric_labels?: string[]
}

export type FieldType = 'text' | 'number' | 'percent' | 'date' | 'boolean' | 'list'

export interface FieldSpec {
  name: string
  type: FieldType
  description?: string
}

export interface Target {
  name: string
  kind: RegionKind
  method: string
  page: number
  bbox: BBox
  instruction?: string
  anchor_heading?: string | null
  header?: string[]
  page_keywords?: string[]
  fields?: FieldSpec[]
  rows?: boolean
}

export interface FieldMatch {
  name: string
  kind: RegionKind
  status: 'matched' | 'ambiguous' | 'not_found'
  margin: number
  best: { page: number; bbox: BBox; score: number; anchor: string; signals: Record<string, number> } | null
  values?: ExtractResult
}

export interface Template {
  id: string
  version: number
  name: string
  source_doc_id: string
  targets: Target[]
  updated_at: string
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`)
  }
  return res.json()
}

export const fileUrl = (docId: string) => `/api/documents/${docId}/file`

// Document metadata never changes for a given sha256, so one request per id is enough.
// App.tsx starts this request while the viewer chunk is still downloading.
const docRequests = new Map<string, Promise<DocMeta>>()

export const api = {
  listDocs: () => fetch('/api/documents').then(json<DocMeta[]>),
  getDoc: (id: string) => {
    let req = docRequests.get(id)
    if (!req) {
      req = fetch(`/api/documents/${id}`).then(json<DocMeta>)
      req.catch(() => docRequests.delete(id))
      docRequests.set(id, req)
    }
    return req
  },
  detectors: () => fetch('/api/detectors').then(json<Detector[]>),
  regions: (id: string, page: number, detector: string) =>
    fetch(`/api/documents/${id}/pages/${page}/regions?detector=${detector}`).then(
      json<PageRegions>,
    ),
  search: (id: string, q: string) =>
    fetch(`/api/documents/${id}/search?q=${encodeURIComponent(q)}`).then(json<SearchHit[]>),
  extract: (
    id: string,
    page: number,
    bbox: BBox,
    kind: RegionKind,
    opts: { fields?: FieldSpec[]; instruction?: string; rows?: boolean } = {},
  ) =>
    fetch(`/api/documents/${id}/extract`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ page, bbox, kind, ...opts }),
    }).then(json<ExtractResult>),
  llm: () => fetch('/api/llm').then(json<{ configured: boolean; client: string | null; model: string | null }>),
  saveTemplate: (body: Omit<Template, 'id' | 'version' | 'updated_at'>, id?: string) =>
    fetch(id ? `/api/templates/${id}` : '/api/templates', {
      method: id ? 'PUT' : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(json<Template>),
  listTemplates: () => fetch('/api/templates').then(json<Template[]>),
  locate: (templateId: string, docId: string) =>
    fetch(`/api/templates/${templateId}/locate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ doc_id: docId }),
    }).then(json<{ fields: FieldMatch[]; version: number }>),
  confirm: (templateId: string, index: number, docId: string, page: number, bbox: BBox) =>
    fetch(`/api/templates/${templateId}/fields/${index}/confirm`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ doc_id: docId, page, bbox }),
    }).then(json<Template>),
}

/** Upload with progress events (fetch can't report upload progress). */
export function uploadPdf(file: File, onProgress: (fraction: number) => void): Promise<DocMeta> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/api/documents')
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total)
    xhr.onload = () => {
      const body = JSON.parse(xhr.responseText || '{}')
      if (xhr.status >= 200 && xhr.status < 300) resolve(body)
      else reject(new Error(body.detail ?? `Upload failed (${xhr.status})`))
    }
    xhr.onerror = () => reject(new Error('Network error during upload'))
    const form = new FormData()
    form.append('file', file)
    xhr.send(form)
  })
}

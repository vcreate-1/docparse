import * as pdfjs from 'pdfjs-dist'
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url'
import { useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import type { PDFDocumentProxy } from 'pdfjs-dist'
import { fileUrl, type BBox, type DocMeta, type Region, type RegionKind, type Target } from '../api'
import { RegionLoader } from '../regionLoader'

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl

const PAGE_GAP = 16

export interface Selection {
  page: number
  bbox: BBox
  kind: RegionKind
  regionId?: string
  /** Which detector the region came from; ids are only unique per detector. */
  detector?: string
  label?: string
}

interface Props {
  doc: DocMeta
  scale: number
  visibleKinds: Set<RegionKind>
  selection: Selection | null
  targets: Target[]
  drawMode: boolean
  jump: { page: number; nonce: number } | null
  onSelect: (s: Selection) => void
  onPageChange: (page: number) => void
  detector: string
  compareDetector: string | null
  onRegionsProgress: (stats: DetectorStats[]) => void
}

export interface DetectorStats {
  detector: string
  loaded: number
  avgMs: number
}

/** One region loader per detector; ML detectors only analyse viewed pages. */
function useLoader(
  docId: string,
  pageCount: number,
  detector: string | null,
): RegionLoader | null {
  const [loader, setLoader] = useState<RegionLoader | null>(null)
  useEffect(() => {
    if (!detector) {
      setLoader(null)
      return
    }
    const l = new RegionLoader(docId, detector)
    setLoader(l)
    // Give on-screen pages a head start before the background sweep begins.
    const t =
      detector === 'structure' ? setTimeout(() => l.prefetchAll(pageCount), 400) : undefined
    return () => {
      clearTimeout(t)
      l.stop()
    }
  }, [docId, pageCount, detector])
  return loader
}

export function Viewer(props: Props) {
  const { doc, scale, jump, onPageChange, onRegionsProgress, detector, compareDetector } = props
  const scroller = useRef<HTMLDivElement>(null)
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Load the PDF lazily: with streaming and auto-fetch off, PDF.js issues
  // range requests for just the pages being rendered.
  useEffect(() => {
    const task = pdfjs.getDocument({
      url: fileUrl(doc.id),
      disableAutoFetch: true,
      disableStream: true,
      rangeChunkSize: 256 * 1024,
    })
    let cancelled = false
    task.promise.then(
      (p) => !cancelled && setPdf(p),
      (e) => !cancelled && setError(String(e?.message ?? e)),
    )
    return () => {
      cancelled = true // a destroyed task rejects with "Loading aborted"; not an error
      setPdf(null)
      task.destroy()
    }
  }, [doc.id])

  const loader = useLoader(doc.id, doc.page_count, detector)
  const compareLoader = useLoader(doc.id, doc.page_count, compareDetector)
  useEffect(() => {
    const loaders = [loader, compareLoader].filter((l): l is RegionLoader => !!l)
    const report = () =>
      onRegionsProgress(
        loaders.map((l) => ({ detector: l.detector, loaded: l.loadedCount, avgMs: l.avgMs })),
      )
    report()
    const unsubs = loaders.map((l) => l.subscribe(report))
    return () => unsubs.forEach((u) => u())
  }, [loader, compareLoader, onRegionsProgress])

  // Cumulative page offsets: O(1) jump-to-page and O(log n) current-page lookup.
  const offsets = useMemo(() => {
    const out: number[] = []
    let y = PAGE_GAP
    for (const [, h] of doc.page_sizes) {
      out.push(y)
      y += h * scale + PAGE_GAP
    }
    out.push(y)
    return out
  }, [doc.page_sizes, scale])

  // Keep the same page in view when zooming.
  const pageBeforeZoom = useRef(1)
  useLayoutEffect(() => {
    const el = scroller.current
    if (el) el.scrollTop = offsets[pageBeforeZoom.current - 1] - PAGE_GAP
  }, [offsets])

  useEffect(() => {
    const el = scroller.current
    if (el && jump) el.scrollTo({ top: offsets[jump.page - 1] - PAGE_GAP })
  }, [jump]) // eslint-disable-line react-hooks/exhaustive-deps -- only on explicit jumps

  useEffect(() => {
    const el = scroller.current
    if (!el) return
    let frame = 0
    const onScroll = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => {
        const mid = el.scrollTop + el.clientHeight / 3
        let lo = 0
        let hi = doc.page_count - 1
        while (lo < hi) {
          const m = (lo + hi + 1) >> 1
          if (offsets[m] <= mid) lo = m
          else hi = m - 1
        }
        pageBeforeZoom.current = lo + 1
        onPageChange(lo + 1)
      })
    }
    el.addEventListener('scroll', onScroll, { passive: true })
    return () => {
      el.removeEventListener('scroll', onScroll)
      cancelAnimationFrame(frame)
    }
  }, [offsets, doc.page_count, onPageChange])

  if (error) return <div className="viewer-error">Could not open PDF: {error}</div>

  return (
    <div className="viewer" ref={scroller}>
      <div className="viewer-inner" style={{ height: offsets[offsets.length - 1] }}>
        {loader && doc.page_sizes.map(([w, h], i) => (
          <PageSlot
            key={i}
            {...props}
            pageNo={i + 1}
            top={offsets[i]}
            width={w * scale}
            height={h * scale}
            pdf={pdf}
            loader={loader}
            compareLoader={compareLoader}
            scrollRoot={scroller}
          />
        ))}
      </div>
    </div>
  )
}

interface SlotProps extends Props {
  pageNo: number
  top: number
  width: number
  height: number
  pdf: PDFDocumentProxy | null
  loader: RegionLoader
  compareLoader: RegionLoader | null
  scrollRoot: React.RefObject<HTMLDivElement | null>
}

function PageSlot(p: SlotProps) {
  const ref = useRef<HTMLDivElement>(null)
  const near = useNearViewport(ref, p.scrollRoot)

  useEffect(() => {
    if (near) {
      p.loader.requestNow(p.pageNo)
      p.compareLoader?.requestNow(p.pageNo)
    }
  }, [near, p.loader, p.compareLoader, p.pageNo])

  return (
    <div
      ref={ref}
      className="page-slot"
      style={{ top: p.top, width: p.width, height: p.height }}
      data-page={p.pageNo}
    >
      {near && p.pdf ? (
        <PageCanvas pdf={p.pdf} pageNo={p.pageNo} width={p.width} height={p.height} />
      ) : (
        <div className="page-placeholder">{p.pageNo}</div>
      )}
      {near && <Overlay {...p} />}
    </div>
  )
}

/** True while the element is within ~1 screen of the visible area. */
function useNearViewport(
  ref: React.RefObject<HTMLElement | null>,
  root: React.RefObject<HTMLElement | null>,
) {
  const [near, setNear] = useState(false)
  useEffect(() => {
    if (!ref.current) return
    const io = new IntersectionObserver(([e]) => setNear(e.isIntersecting), {
      root: root.current,
      rootMargin: '100% 0px',
    })
    io.observe(ref.current)
    return () => io.disconnect()
  }, [ref, root])
  return near
}

function PageCanvas({
  pdf,
  pageNo,
  width,
  height,
}: {
  pdf: PDFDocumentProxy
  pageNo: number
  width: number
  height: number
}) {
  const canvas = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    let cancelled = false
    let task: ReturnType<pdfjs.PDFPageProxy['render']> | null = null
    // Debounce so a zoom drag doesn't start a render per intermediate scale.
    const timer = setTimeout(async () => {
      const page = await pdf.getPage(pageNo)
      if (cancelled || !canvas.current) return
      const base = page.getViewport({ scale: 1 })
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      const viewport = page.getViewport({ scale: (width / base.width) * dpr })
      // Render offscreen, then swap, so the old bitmap stays visible meanwhile.
      const off = document.createElement('canvas')
      off.width = Math.floor(viewport.width)
      off.height = Math.floor(viewport.height)
      task = page.render({ canvas: off, viewport })
      try {
        await task.promise
      } catch {
        return // cancelled
      }
      const c = canvas.current
      if (cancelled || !c) return
      c.width = off.width
      c.height = off.height
      c.getContext('2d')?.drawImage(off, 0, 0)
    }, 30)
    return () => {
      cancelled = true
      clearTimeout(timer)
      task?.cancel()
    }
  }, [pdf, pageNo, width])

  return <canvas ref={canvas} className="page-canvas" style={{ width, height }} />
}

function Overlay(p: SlotProps) {
  const regions = useSyncExternalStore(
    (fn) => p.loader.subscribe(fn),
    () => p.loader.get(p.pageNo),
  )
  const compareRegions = useSyncExternalStore(
    (fn) => p.compareLoader?.subscribe(fn) ?? (() => {}),
    () => p.compareLoader?.get(p.pageNo),
  )
  const [draft, setDraft] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(
    null,
  )

  const point = (e: React.PointerEvent) => {
    const r = (e.currentTarget as SVGElement).getBoundingClientRect()
    return {
      x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
      y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
    }
  }

  const [hovered, setHovered] = useState<{ region: Region; detector: string } | null>(null)

  const select = (region: Region, detector: string) =>
    p.onSelect({
      page: p.pageNo,
      bbox: region.bbox,
      kind: region.kind,
      regionId: region.id,
      detector,
      label: region.label,
    })

  const sel = p.selection?.page === p.pageNo ? p.selection : null
  const isSelected = (r: Region, detector: string) =>
    sel?.regionId === r.id && sel.detector === detector
  const pageTargets = p.targets.filter((t) => t.page === p.pageNo)
  // A region already saved as a target is drawn in the target style instead
  // of stacking a second outline on top of it.
  const targetFor = (r: Region) => pageTargets.find((t) => sameBox(t.bbox, r.bbox))
  const selectedRegion =
    sel && ([...(regions?.regions ?? []), ...(compareRegions?.regions ?? [])].find(
      (r) => sel.regionId === r.id,
    ) ?? null)

  const regionRects = (list: Region[] | undefined, detector: string, compare: boolean) =>
    list
      ?.filter((r) => p.visibleKinds.has(r.kind))
      .map((r) => {
        const target = targetFor(r)
        const selected = isSelected(r, detector)
        return (
          <rect
            key={`${detector}-${r.id}`}
            className={[
              'region',
              `kind-${r.kind}`,
              compare && !selected ? 'compare' : '',
              selected ? 'selected' : '',
              target ? 'in-template' : '',
            ].join(' ')}
            x={r.bbox[0]}
            y={r.bbox[1]}
            width={r.bbox[2] - r.bbox[0]}
            height={r.bbox[3] - r.bbox[1]}
            vectorEffect="non-scaling-stroke"
            tabIndex={p.drawMode ? -1 : 0}
            role="button"
            aria-pressed={selected}
            aria-label={`${KIND_LABELS[r.kind]}: ${r.label}, page ${p.pageNo}${target ? `, in template as ${target.name}` : ''}`}
            onClick={() => select(r, detector)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                select(r, detector)
              }
            }}
            onPointerEnter={() => setHovered({ region: r, detector })}
            onPointerLeave={() => setHovered((h) => (h?.region === r ? null : h))}
            onFocus={() => setHovered({ region: r, detector })}
            onBlur={() => setHovered((h) => (h?.region === r ? null : h))}
          />
        )
      })

  // The label tag: instant (unlike a native tooltip) and the non-colour cue
  // for what each box is. Shown for the hovered box, else the selected one.
  const tagged =
    hovered ??
    (selectedRegion && sel?.detector ? { region: selectedRegion, detector: sel.detector } : null)

  return (
    <>
      <svg
        className={`overlay ${p.drawMode ? 'drawing' : ''} ${sel ? 'has-sel' : ''}`}
        viewBox="0 0 1 1"
        preserveAspectRatio="none"
        onPointerDown={(e) => {
          if (!p.drawMode) return
          const { x, y } = point(e)
          ;(e.currentTarget as SVGElement).setPointerCapture(e.pointerId)
          setDraft({ x0: x, y0: y, x1: x, y1: y })
        }}
        onPointerMove={(e) => {
          if (!draft) return
          const { x, y } = point(e)
          setDraft({ ...draft, x1: x, y1: y })
        }}
        onPointerUp={() => {
          if (!draft) return
          const bbox: BBox = [
            Math.min(draft.x0, draft.x1),
            Math.min(draft.y0, draft.y1),
            Math.max(draft.x0, draft.x1),
            Math.max(draft.y0, draft.y1),
          ]
          setDraft(null)
          if (bbox[2] - bbox[0] > 0.01 && bbox[3] - bbox[1] > 0.005) {
            p.onSelect({ page: p.pageNo, bbox, kind: 'custom', label: 'Custom region' })
          }
        }}
      >
        {sel && (
          // Dims the page around the selection so it reads as the one thing in focus.
          <path
            className="scrim"
            fillRule="evenodd"
            d={`M0 0H1V1H0Z M${sel.bbox[0]} ${sel.bbox[1]}V${sel.bbox[3]}H${sel.bbox[2]}V${sel.bbox[1]}Z`}
          />
        )}
        {regionRects(regions?.regions, p.detector, false)}
        {p.compareDetector && regionRects(compareRegions?.regions, p.compareDetector, true)}
        {pageTargets
          .filter((t) => !regions?.regions.some((r) => sameBox(t.bbox, r.bbox) && p.visibleKinds.has(r.kind)))
          .map((t, i) => (
            <rect
              key={`t${i}`}
              className="target"
              x={t.bbox[0]}
              y={t.bbox[1]}
              width={t.bbox[2] - t.bbox[0]}
              height={t.bbox[3] - t.bbox[1]}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        {sel && !sel.regionId && (
          <rect
            className="region kind-custom selected"
            x={sel.bbox[0]}
            y={sel.bbox[1]}
            width={sel.bbox[2] - sel.bbox[0]}
            height={sel.bbox[3] - sel.bbox[1]}
            vectorEffect="non-scaling-stroke"
          />
        )}
        {draft && (
          <rect
            className="region kind-custom drafting"
            x={Math.min(draft.x0, draft.x1)}
            y={Math.min(draft.y0, draft.y1)}
            width={Math.abs(draft.x1 - draft.x0)}
            height={Math.abs(draft.y1 - draft.y0)}
            vectorEffect="non-scaling-stroke"
          />
        )}
      </svg>
      {pageTargets.map((t, i) => (
        <span
          key={`tt${i}`}
          className={`region-tag target-tag ${t.bbox[3] > 0.97 ? 'inside' : ''}`}
          style={{ left: `${t.bbox[0] * 100}%`, top: `${t.bbox[3] * 100}%` }}
        >
          {t.name}
        </span>
      ))}
      {tagged && !p.drawMode && (
        <span
          className={`region-tag kind-${tagged.region.kind} ${tagged.region.bbox[1] < 0.03 ? 'inside' : ''}`}
          style={{
            left: `${tagged.region.bbox[0] * 100}%`,
            top: `${tagged.region.bbox[1] * 100}%`,
          }}
        >
          {KIND_LABELS[tagged.region.kind]}
          {tagged.region.title ? ` · ${tagged.region.title}` : ''}
          {tagged.region.score !== undefined ? ` · ${Math.round(tagged.region.score * 100)}%` : ''}
          {p.compareDetector && <span className="tag-source"> {tagged.detector}</span>}
        </span>
      )}
    </>
  )
}

function sameBox(a: BBox, b: BBox) {
  return a.every((v, i) => Math.abs(v - b[i]) < 0.002)
}

export const KIND_LABELS: Record<RegionKind, string> = {
  table: 'Table',
  chart_vector: 'Chart',
  image: 'Image',
  heading: 'Heading',
  text: 'Text',
  scanned_page: 'Scanned',
  custom: 'Custom region',
}

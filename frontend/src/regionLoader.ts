import { api, type PageRegions } from './api'

/**
 * Fetches per-page region detection with two priority lanes:
 * pages on screen jump the queue; everything else is prefetched behind them.
 * Each page is requested at most once per document.
 */
export class RegionLoader {
  private results = new Map<number, PageRegions>()
  private queued = new Set<number>()
  private urgent: number[] = []
  private background: number[] = []
  private active = 0
  private listeners = new Set<() => void>()
  private stopped = false
  private totalMs = 0
  private readonly docId: string
  readonly detector: string
  private readonly concurrency: number

  constructor(docId: string, detector: string, concurrency = 4) {
    this.docId = docId
    this.detector = detector
    // ML detectors take ~1 s/page on CPU; don't flood the server with them.
    this.concurrency = detector === 'structure' ? concurrency : 2
  }

  get(page: number): PageRegions | undefined {
    return this.results.get(page)
  }

  get loadedCount(): number {
    return this.results.size
  }

  /** Average server-side detection time per page, for comparing detectors. */
  get avgMs(): number {
    return this.results.size ? this.totalMs / this.results.size : 0
  }

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn)
    return () => this.listeners.delete(fn)
  }

  /** A page became visible: fetch it before anything in the background lane. */
  requestNow(page: number) {
    if (this.results.has(page)) return
    this.background = this.background.filter((p) => p !== page)
    if (!this.urgent.includes(page)) this.urgent.unshift(page)
    this.queued.add(page)
    this.pump()
  }

  prefetchAll(pageCount: number) {
    for (let p = 1; p <= pageCount; p++) {
      if (!this.queued.has(p) && !this.results.has(p)) {
        this.background.push(p)
        this.queued.add(p)
      }
    }
    this.pump()
  }

  stop() {
    this.stopped = true
    this.listeners.clear()
  }

  private pump() {
    // Background work only uses half the slots, so a page scrolled into view
    // never waits behind a full queue of prefetches.
    while (!this.stopped && this.active < this.concurrency) {
      const page =
        this.urgent.shift() ??
        (this.active < this.concurrency / 2 ? this.background.shift() : undefined)
      if (page === undefined) return
      this.active++
      api
        .regions(this.docId, page, this.detector)
        .then((r) => {
          this.results.set(page, r)
          this.totalMs += r.ms
          this.listeners.forEach((fn) => fn())
        })
        .catch(() => this.queued.delete(page)) // allow a retry on next view
        .finally(() => {
          this.active--
          this.pump()
        })
    }
  }
}

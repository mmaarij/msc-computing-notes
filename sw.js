const CACHE = "notes-reader-v13"

const SHELL = [
  "./",
  "./index.html",
  "./manifest.json",
  "./icon.svg",
  "./README.md",
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js",
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js",
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf_viewer.min.css"
]

const PDFS = [
  "./Machine Learning/Compiled Notes.pdf",
  "./Text and Sequence Analytics/Compiled Notes.pdf",
  "./Statistical Computing/Compiled Notes.pdf",
  "./Statistical Computing/Lab Notes.pdf",
  "./Cybersecurity and Secure Programming/Compiled Notes.pdf"
]

function getCanonicalUrl(requestUrl) {
  const url = new URL(requestUrl)
  if (url.origin === self.location.origin) {
    url.search = ""
    url.hash = ""
  }
  return url.toString()
}

self.addEventListener("install", e => {
  e.waitUntil(
    caches.open(CACHE).then(async cache => {
      await cache.addAll(SHELL)
      // Pre-cache PDFs best-effort during install so offline works immediately
      await Promise.allSettled(
        PDFS.map(async path => {
          const reqUrl = new URL(path, self.location.href).toString()
          const res = await fetch(reqUrl, { cache: "no-cache" })
          if (res && res.ok) {
            await cache.put(reqUrl, res)
          }
        })
      )
    })
  )
  self.skipWaiting()
})

self.addEventListener("activate", e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  )
  self.clients.claim()
})

self.addEventListener("fetch", e => {
  if (e.request.method !== "GET") return

  const url = new URL(e.request.url)
  const isSameOrigin = url.origin === self.location.origin
  const isDynamicContent =
    isSameOrigin &&
    (url.pathname.endsWith(".pdf") ||
      url.pathname.endsWith("README.md") ||
      url.pathname.endsWith("index.html") ||
      url.pathname.endsWith("manifest.json") ||
      url.pathname.endsWith("/"))

  // PDFs, README, and app entry: network-first (bypassing stale HTTP cache) so online launches
  // always fetch and cache the latest version, falling back to cached copy when offline.
  if (isDynamicContent) {
    const canonicalKey = getCanonicalUrl(e.request.url)
    e.respondWith(
      fetch(e.request, { cache: "no-cache" })
        .then(res => {
          if (res && res.ok && res.status === 200) {
            const clone = res.clone()
            caches.open(CACHE).then(c => c.put(canonicalKey, clone))
          }
          return res
        })
        .catch(async () => {
          const cached =
            (await caches.match(canonicalKey, { ignoreSearch: true })) ||
            (await caches.match(e.request, { ignoreSearch: true }))
          if (cached) return cached
          if (url.pathname.endsWith("/")) {
            return caches.match(new URL("./index.html", self.location.href).toString(), {
              ignoreSearch: true
            })
          }
          return Response.error()
        })
    )
    return
  }

  // Static shell, PDF.js worker, and Google Fonts: cache-first, caching any newly fetched assets for offline use
  e.respondWith(
    caches.match(e.request).then(cached => {
      if (cached) return cached
      return fetch(e.request)
        .then(res => {
          if (res && (res.ok || res.type === "opaque")) {
            const clone = res.clone()
            caches.open(CACHE).then(c => c.put(e.request, clone))
          }
          return res
        })
        .catch(() => caches.match(e.request, { ignoreSearch: true }))
    })
  )
})

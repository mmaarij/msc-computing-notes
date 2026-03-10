const CACHE = "notes-reader-v2"

const SHELL = [
  "./index.html",
  "./icon.svg",
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"
]

self.addEventListener("install", e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(SHELL))
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
  const url = new URL(e.request.url)

  // PDFs, README, and index.html: network-first so updates always come through
  if (url.pathname.endsWith(".pdf") || url.pathname.endsWith("README.md") || url.pathname.endsWith("index.html") || url.pathname.endsWith("/")) {
    e.respondWith(
      fetch(e.request).then(res => {
        const clone = res.clone()
        caches.open(CACHE).then(c => c.put(e.request, clone))
        return res
      }).catch(() => caches.match(e.request))
    )
    return
  }

  // Shell assets: cache-first
  e.respondWith(
    caches.match(e.request).then(r => r || fetch(e.request))
  )
})

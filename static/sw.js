const CACHE_NAME = 'plant-detect-v1';
const URLs_TO_CACHE = [
  '/',
  '/static/css/style.css',
  '/static/js/app.js',
  '/static/manifest.json',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png',
  // add other static assets if needed
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(URLs_TO_CACHE))
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.url.includes('/predict') || req.url.includes('/health')) {
    event.respondWith(
      fetch(req).catch(() => caches.match('/'))
    );
    return;
  }
  // For static assets: cache-first
  event.respondWith(
    caches.match(req).then(cached => cached || fetch(req))
  );
});

self.addEventListener('activate', (event) => {
  // cleanup old caches
  event.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
    ))
  );
});

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

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open('plant-disease-v1').then(cache => {
      // addAll is all-or-nothing, so use individual adds instead

      return Promise.allSettled(
        URLs_TO_CACHE.map(url =>
          cache.add(url).catch(err =>
            console.warn(`Failed to cache ${url}:`, err)
          )
        )
      );
    })
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

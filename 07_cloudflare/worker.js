const TRIMBLE_MASTER = 'https://app.connect.trimble.com/tc/api/2.0';
const regionCache = new Map();

addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request));
});

async function handleRequest(request) {
  if (request.method === 'OPTIONS') return corsWrap(new Response(null, { status: 204 }));

  const url = new URL(request.url);

  if (url.pathname === '/ping') return corsWrap(new Response('pong', { status: 200 }));

  const isApi  = url.pathname.startsWith('/api/');
  const isWopi = url.pathname.startsWith('/wopi/');
  if (!isApi && !isWopi) {
    return corsWrap(new Response('Not found', { status: 404 }));
  }

  const auth = request.headers.get('Authorization');
  if (!auth || !auth.startsWith('Bearer ')) {
    return corsWrap(new Response('Unauthorized', { status: 401 }));
  }

  const region    = (url.searchParams.get('region') || 'europe').toLowerCase();
  const tcApiBase = await resolveRegionUrl(auth, region);

  const trimblePath   = isApi ? url.pathname.replace('/api', '') : url.pathname;
  const forwardParams = new URLSearchParams(url.search);
  forwardParams.delete('region');
  const qs         = forwardParams.toString();
  const trimbleUrl = tcApiBase + trimblePath + (qs ? '?' + qs : '');

  // --- Headers opbouwen ---
  const headers = new Headers();
  headers.set('Authorization', auth);
  headers.set('Accept', request.headers.get('Accept') || 'application/json');

  const ct = request.headers.get('Content-Type') || '';
  // multipart/form-data NIET doorsturen: de boundary zit embedded in de body stream.
  // Cloudflare herstelt de boundary automatisch als je hem weglaat.
  // Alle andere content-types wel doorsturen.
  if (ct && !ct.toLowerCase().includes('multipart')) {
    headers.set('Content-Type', ct);
  }

  // --- Request doorsturen ---
  const isDownload = trimblePath.includes('/binarydata') || trimblePath.includes('/transfer');
  const hasBody    = !['GET', 'HEAD'].includes(request.method);

  let upstream;
  try {
    upstream = await fetch(trimbleUrl, {
      method:   request.method,
      headers:  headers,
      body:     hasBody ? request.body : undefined,
      redirect: isDownload ? 'manual' : 'follow',
    });
  } catch (e) {
    return corsWrap(new Response('Upstream fout: ' + e.message, { status: 502 }));
  }

  // Redirect onderscheppen voor download endpoints
  if (isDownload && [301, 302, 303].includes(upstream.status)) {
    const location = upstream.headers.get('location');
    return corsWrap(new Response(JSON.stringify({ downloadUrl: location }), {
      status:  200,
      headers: { 'Content-Type': 'application/json' },
    }));
  }

  const body        = await upstream.arrayBuffer();
  const contentType = upstream.headers.get('Content-Type') || 'application/octet-stream';
  return corsWrap(new Response(body, {
    status:  upstream.status,
    headers: { 'Content-Type': contentType },
  }));
}

async function resolveRegionUrl(auth, regionLabel) {
  const key = regionLabel.toLowerCase().trim();
  if (regionCache.has(key)) return regionCache.get(key);

  try {
    const resp = await fetch(TRIMBLE_MASTER + '/regions', {
      headers: { 'Authorization': auth, 'Accept': 'application/json' }
    });
    if (resp.ok) {
      const list = await resp.json();
      for (const r of list) {
        if (r['tc-api']) {
          regionCache.set((r.location || '').toLowerCase(), r['tc-api'].replace(/\/$/, ''));
        }
      }
    }
  } catch (e) {}

  return regionCache.get(key) || TRIMBLE_MASTER;
}

function corsWrap(response) {
  const r = new Response(response.body, response);
  r.headers.set('Access-Control-Allow-Origin',  '*');
  r.headers.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  r.headers.set('Access-Control-Allow-Headers', 'Authorization, Content-Type, Accept');
  r.headers.set('Access-Control-Max-Age',       '86400');
  return r;
}
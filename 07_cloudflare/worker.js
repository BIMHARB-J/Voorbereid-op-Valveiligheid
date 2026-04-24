/**
 * EdgeProtector — Cloudflare Worker (CORS proxy voor Trimble REST API)
 *
 * Deploy stappen:
 *  1. Ga naar https://workers.cloudflare.com en maak een gratis account
 *  2. Klik "Create Worker"
 *  3. Plak deze hele file in de editor
 *  4. Klik "Save and Deploy"
 *  5. Kopieer de worker-URL (bijv. https://edgeprotector.jouwnaam.workers.dev)
 *  6. Plak die URL als PROXY_BASE in edgeprotector.html
 */

const ALLOWED_ORIGIN_CONTAINS = 'connect.trimble.com';
const TRIMBLE_BASE = 'https://app.eu.connect.trimble.com/tc/api/2.0';

export default {
  async fetch(request, env, ctx) {

    // OPTIONS preflight — altijd toestaan
    if (request.method === 'OPTIONS') {
      return corsResponse(new Response(null, { status: 204 }));
    }

    const url = new URL(request.url);

    // Health check: GET /ping
    if (url.pathname === '/ping') {
      return corsResponse(new Response('pong', { status: 200 }));
    }

    // Alleen /api/* paden proxyen
    if (!url.pathname.startsWith('/api/')) {
      return corsResponse(new Response('Not found', { status: 404 }));
    }

    // Authorization header verplicht — de extensie stuurt het Bearer token mee
    const auth = request.headers.get('Authorization');
    if (!auth || !auth.startsWith('Bearer ')) {
      return corsResponse(new Response('Unauthorized', { status: 401 }));
    }

    // Bouw de Trimble doel-URL
    // /api/projects/XYZ/files → TRIMBLE_BASE/projects/XYZ/files
    const trimblePath = url.pathname.replace('/api/', '/');
    const trimbleUrl  = TRIMBLE_BASE + trimblePath + url.search;

    // Stuur het verzoek door naar Trimble
    const proxyReq = new Request(trimbleUrl, {
      method:  request.method,
      headers: {
        'Authorization': auth,
        'Accept':        'application/json',
        'Content-Type':  request.headers.get('Content-Type') || 'application/json',
      },
      body: ['GET', 'HEAD'].includes(request.method) ? undefined : request.body,
    });

    let trimbleResp;
    try {
      trimbleResp = await fetch(proxyReq);
    } catch (e) {
      return corsResponse(new Response(`Upstream error: ${e.message}`, { status: 502 }));
    }

    // Antwoord doorsturen inclusief CORS headers
    const respBody    = await trimbleResp.arrayBuffer();
    const respHeaders = new Headers({
      'Content-Type':  trimbleResp.headers.get('Content-Type') || 'application/octet-stream',
      'X-Proxy-Status': trimbleResp.status,
    });

    return corsResponse(new Response(respBody, {
      status:  trimbleResp.status,
      headers: respHeaders,
    }));
  }
};

function corsResponse(response) {
  const r = new Response(response.body, response);
  r.headers.set('Access-Control-Allow-Origin',  '*');
  r.headers.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  r.headers.set('Access-Control-Allow-Headers', 'Authorization, Content-Type, Accept');
  r.headers.set('Access-Control-Max-Age',       '86400');
  return r;
}
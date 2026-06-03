"""
Valwijzer Flask Server — async job systeem
==========================================
De analyse van grote IFC bestanden kan minuten duren. Om timeouts te
voorkomen werkt de server asynchroon:

  POST /analyse          → start job, geeft { job_id } terug
  GET  /status/{job_id}  → { status: queued|running|done|error, message }
  GET  /result/{job_id}  → download het gegenereerde IFC (als status=done)
  GET  /health           → server status
"""

import os
import sys
import uuid
import tempfile
import traceback
import threading
from pathlib import Path
from flask import Flask, request, send_file, jsonify, make_response

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

# In-memory job store: { job_id: { status, message, output_path, output_name } }
jobs = {}
jobs_lock = threading.Lock()

# ── CORS ──────────────────────────────────────────────────────────
CORS_HEADERS = {
    'Access-Control-Allow-Origin':  '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, Accept',
    'Access-Control-Max-Age':       '86400',
}

@app.after_request
def after(response):
    for k, v in CORS_HEADERS.items():
        response.headers[k] = v
    return response

@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def preflight(path):
    r = make_response('', 204)
    for k, v in CORS_HEADERS.items():
        r.headers[k] = v
    return r

# ── Health ────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return jsonify({
        'status': 'ok',
        'service': 'Valwijzer analyse server',
        'files': os.listdir(script_dir),
        'active_jobs': len(jobs),
    })

# ── Start analyse job ─────────────────────────────────────────────
@app.route('/analyse', methods=['POST'])
def analyse():
    download_url  = request.form.get('download_url')
    original_name = request.form.get('filename', 'model.ifc')

    if not download_url and 'ifc' not in request.files:
        return jsonify({'error': 'Geef download_url of ifc bestand mee'}), 400

    job_id   = str(uuid.uuid4())
    tmp_dir  = tempfile.mkdtemp()   # blijft bestaan na het verzoek
    stem     = Path(original_name).stem
    out_name = f'{stem}_valgevaren.ifc'
    in_path  = os.path.join(tmp_dir, 'input.ifc')
    out_path = os.path.join(tmp_dir, out_name)

    # Sla het bestand alvast op als het direct geüpload werd
    if 'ifc' in request.files:
        request.files['ifc'].save(in_path)

    with jobs_lock:
        jobs[job_id] = {
            'status':      'queued',
            'message':     'In wachtrij',
            'output_path': out_path,
            'output_name': out_name,
        }

    # Start de analyse in een achtergrond-thread
    t = threading.Thread(
        target=_run_job,
        args=(job_id, in_path, out_path, stem, download_url),
        daemon=True,
    )
    t.start()

    return jsonify({'job_id': job_id}), 202


# ── Job status opvragen ───────────────────────────────────────────
@app.route('/status/<job_id>', methods=['GET'])
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Onbekende job'}), 404
    return jsonify({
        'status':  job['status'],
        'message': job['message'],
    })


# ── Resultaat downloaden ──────────────────────────────────────────
@app.route('/result/<job_id>', methods=['GET'])
def result(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Onbekende job'}), 404
    if job['status'] != 'done':
        return jsonify({'error': f'Job nog niet klaar (status: {job["status"]})'}), 409
    if not os.path.exists(job['output_path']):
        return jsonify({'error': 'Resultaatbestand niet gevonden'}), 500

    return send_file(
        job['output_path'],
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name=job['output_name'],
    )


# ── Achtergrond job uitvoeren ─────────────────────────────────────
def _run_job(job_id, in_path, out_path, stem, download_url):
    def update(status, message):
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]['status']  = status
                jobs[job_id]['message'] = message

    try:
        # Stap 1: bestand ophalen (als download_url meegegeven)
        if download_url:
            update('running', 'Bestand ophalen van Trimble...')
            import urllib.request
            req = urllib.request.Request(download_url)
            with urllib.request.urlopen(req, timeout=180) as resp:
                with open(in_path, 'wb') as f:
                    f.write(resp.read())

        # Stap 2: analyse uitvoeren
        update('running', 'Analyse uitvoeren...')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        import analyse
        models = [(stem, in_path)]
        analyse.build(models, out_path)

        if not os.path.exists(out_path):
            update('error', 'Script heeft geen uitvoer gegenereerd')
            return

        update('done', 'Klaar')

    except Exception as e:
        traceback.print_exc()
        update('error', str(e))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
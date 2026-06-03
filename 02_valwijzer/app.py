"""
Valwijzer Flask Server
======================
Ontvangt een download_url van Trimble, downloadt het IFC zelf,
voert het analyse script uit als subprocess, en geeft het
gegenereerde IFC terug als download.
"""

import os
import sys
import tempfile
import traceback
import subprocess
from pathlib import Path
from flask import Flask, request, send_file, jsonify, make_response

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return jsonify({
        'status': 'ok',
        'service': 'Valwijzer analyse server',
        'files': os.listdir(script_dir),
    })

# ── Analyse ───────────────────────────────────────────────────────
@app.route('/analyse', methods=['POST'])
def analyse():
    download_url  = request.form.get('download_url')
    original_name = request.form.get('filename', 'model.ifc')

    if not download_url and 'ifc' not in request.files:
        return jsonify({'error': 'Geef download_url of ifc bestand mee'}), 400

    stem     = Path(original_name).stem
    out_name = f'{stem}_valgevaren.ifc'

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path  = os.path.join(tmpdir, 'input.ifc')
        out_path = os.path.join(tmpdir, out_name)

        # Bestand ophalen
        if download_url:
            import urllib.request
            try:
                with urllib.request.urlopen(download_url, timeout=300) as resp:
                    with open(in_path, 'wb') as f:
                        f.write(resp.read())
            except Exception as e:
                return jsonify({'error': f'Download mislukt: {str(e)}'}), 500
        else:
            request.files['ifc'].save(in_path)

        size_mb = os.path.getsize(in_path) / (1024 * 1024)
        print(f'Bestand ontvangen: {size_mb:.1f} MB — analyse starten...')

        # Analyse als subprocess (geheugen vrijgegeven na afloop)
        script_dir    = os.path.dirname(os.path.abspath(__file__))
        analyse_script = os.path.join(script_dir, 'analyse.py')

        try:
            proc = subprocess.run(
                [sys.executable, analyse_script, in_path, '-o', out_path],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=script_dir,
            )
        except subprocess.TimeoutExpired:
            return jsonify({'error': 'Analyse timeout (>10 minuten)'}), 500
        except Exception as e:
            return jsonify({'error': f'Subprocess fout: {str(e)}'}), 500

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or 'Onbekende fout')[-500:]
            print(f'Script fout:\n{err}')
            return jsonify({'error': f'Analyse mislukt: {err}'}), 500

        if not os.path.exists(out_path):
            return jsonify({'error': 'Script heeft geen uitvoer gegenereerd'}), 500

        print(f'Analyse klaar: {out_name}')
        return send_file(
            out_path,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=out_name,
        )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
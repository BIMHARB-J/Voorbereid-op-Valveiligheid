"""
Valwijzer Flask Server
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path
from flask import Flask, request, send_file, jsonify, make_response

app = Flask(__name__)

# ── CORS: handmatig op elke response ──────────────────────────────
# flask-cors werkt soms niet correct voor multipart uploads vanuit
# een browser-iframe. We zetten de headers handmatig op elke response.
CORS_HEADERS = {
    'Access-Control-Allow-Origin':  '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, Accept',
    'Access-Control-Max-Age':       '86400',
}

def cors(response):
    for k, v in CORS_HEADERS.items():
        response.headers[k] = v
    return response

@app.after_request
def after(response):
    return cors(response)

# OPTIONS preflight voor alle routes
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def preflight(path):
    return cors(make_response('', 204))

# Maximale bestandsgrootte: 200 MB
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024


@app.route('/health', methods=['GET'])
def health():
    import os, sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(script_dir)
    return jsonify({
        'status': 'ok',
        'service': 'Valwijzer analyse server',
        'script_dir': script_dir,
        'files_in_dir': files,
        'sys_path': sys.path[:5],
    })


@app.route('/analyse', methods=['POST'])
def analyse():
    """
    Twee modi:
    A) download_url + token meesturen → server downloadt zelf van Trimble
    B) ifc bestand direct uploaden (voor kleine bestanden)
    """
    import urllib.request

    original_name = request.form.get('filename', 'model.ifc')
    stem = Path(original_name).stem
    output_name = f'{stem}_valgevaren.ifc'

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path  = os.path.join(tmpdir, 'input.ifc')
        output_path = os.path.join(tmpdir, output_name)

        # Modus A: server downloadt zelf via de pre-signed URL
        download_url = request.form.get('download_url')
        if download_url:
            try:
                req = urllib.request.Request(download_url)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    with open(input_path, 'wb') as f:
                        f.write(resp.read())
            except Exception as e:
                return jsonify({'error': f'Download van Trimble mislukt: {str(e)}'}), 500

        # Modus B: bestand direct geüpload
        elif 'ifc' in request.files:
            request.files['ifc'].save(input_path)

        else:
            return jsonify({'error': 'Geef download_url of ifc bestand mee'}), 400

        try:
            _run_analysis(input_path, output_path, stem)
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'Analyse mislukt: {str(e)}'}), 500

        if not os.path.exists(output_path):
            return jsonify({'error': 'Analyse script heeft geen uitvoer gegenereerd'}), 500

        return send_file(
            output_path,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=output_name,
        )


def _run_analysis(input_path: str, output_path: str, label: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import analyse
    models = [(label, input_path)]
    analyse.build(models, output_path)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
"""
Valwijzer Flask Server
======================
Ontvangt een IFC bestand, voert het valgevaar-analyse script uit,
en geeft het gegenereerde IFC terug als download.

Endpoints:
  POST /analyse   - IFC bestand uploaden, geanalyseerd IFC terugkrijgen
  GET  /health    - server status check
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # sta cross-origin requests toe vanuit de Trimble extensie

# Maximale bestandsgrootte: 200 MB
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'Valwijzer analyse server'})


@app.route('/analyse', methods=['POST'])
def analyse():
    """
    Verwacht:
      - multipart/form-data met veld 'ifc' (het IFC bestand)
      - optioneel veld 'filename' (originele bestandsnaam)

    Geeft terug:
      - het gegenereerde IFC bestand als download
    """
    if 'ifc' not in request.files:
        return jsonify({'error': 'Geen IFC bestand meegestuurd (veld: ifc)'}), 400

    ifc_file = request.files['ifc']
    original_name = request.form.get('filename', ifc_file.filename or 'model.ifc')

    # Bepaal output naam
    stem = Path(original_name).stem
    output_name = f'{stem}_valgevaren.ifc'

    # Werk in een tijdelijke map zodat parallelle requests niet conflicteren
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path  = os.path.join(tmpdir, 'input.ifc')
        output_path = os.path.join(tmpdir, output_name)

        # Sla het inkomende bestand op
        ifc_file.save(input_path)

        try:
            # Voer het analyse script uit
            _run_analysis(input_path, output_path, stem)
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'Analyse mislukt: {str(e)}'}), 500

        if not os.path.exists(output_path):
            return jsonify({'error': 'Analyse script heeft geen uitvoer gegenereerd'}), 500

        # Stuur het gegenereerde IFC terug
        return send_file(
            output_path,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=output_name,
        )


def _run_analysis(input_path: str, output_path: str, label: str):
    """
    Roept het valwijzer analyse script aan.
    Importeert het script direct als module zodat er geen subprocess nodig is.
    """
    # Importeer het analyse script (in dezelfde map)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    import analyse  # analyse.py = het valwijzer script

    models = [(label, input_path)]
    analyse.build(models, output_path)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)

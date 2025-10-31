"""
CorBas Backend - NO CACHE VERSION
===================================
Forces browser to always load fresh content
"""

from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
import spacy
import os
import io

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

HAS_PYMUSAS = False
HAS_PYMUPDF = False

# Try PyMUSAS
try:
    from pymusas.rankers.lexicon_entry import ContextualRuleBasedRanker
    from pymusas.taggers.rules.single_word import SingleWordRule
    from pymusas.taggers.rules.mwe import MWERule
    from pymusas.pos_mapper import UPOS_TO_USAS_CORE
    import pymusas.lexicon_collection
    
    lexicon_lookup = pymusas.lexicon_collection.LexiconCollection.from_tsv(None, include_pos=True)
    ranker = ContextualRuleBasedRanker(*ContextualRuleBasedRanker.get_construction_arguments(lexicon_lookup))
    single_word_rule = SingleWordRule(lexicon_lookup, ranker)
    mwe_rule = MWERule(lexicon_lookup, ranker)
    
    nlp.add_pipe('pymusas_rule_based_tagger', config={
        "rules": [mwe_rule, single_word_rule],
        "ranker": ranker,
        "pos_mapper": UPOS_TO_USAS_CORE
    }, last=True)
    HAS_PYMUSAS = True
    print("✓ PyMUSAS loaded")
except Exception as e:
    print(f"⚠ PyMUSAS failed: {e}")

# Try PyMuPDF
try:
    import fitz
    HAS_PYMUPDF = True
    print("✓ PyMuPDF loaded")
except:
    print("⚠ PyMuPDF not installed")

print(f"✓ Backend ready! Pipes: {nlp.pipe_names}")


@app.route('/')
def index():
    """Serve HTML with NO CACHE headers"""
    try:
        html_path = os.path.join(BASE_DIR, 'corbas.html')
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Force correct backend URL
        backend_url = f"{request.scheme}://{request.host}"
        content = content.replace(
            "const BACKEND_URL = 'http://127.0.0.1:5000';",
            f"const BACKEND_URL = '{backend_url}';"
        )
        
        # CRITICAL: Add no-cache headers
        response = make_response(content)
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        print(f"✓ Serving HTML with backend URL: {backend_url}")
        return response
        
    except Exception as e:
        return f"<h1>Error</h1><p>{str(e)}</p><p>Path: {BASE_DIR}</p>", 500


@app.route('/service-worker.js')
def service_worker():
    """Return a service worker that UNREGISTERS itself"""
    sw_code = """
// Unregister service worker
self.addEventListener('install', () => {
    self.skipWaiting();
});

self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(names => {
            return Promise.all(names.map(name => caches.delete(name)));
        }).then(() => {
            return self.registration.unregister();
        }).then(() => {
            return self.clients.matchAll();
        }).then(clients => {
            clients.forEach(client => client.navigate(client.url));
        })
    );
});

console.log('Service Worker: UNREGISTERING...');
"""
    response = make_response(sw_code)
    response.headers['Content-Type'] = 'application/javascript'
    response.headers['Cache-Control'] = 'no-store'
    return response


@app.route('/manifest.json')
def manifest():
    """Basic manifest"""
    return jsonify({
        "name": "CorBas",
        "short_name": "CorBas",
        "start_url": "/",
        "display": "standalone"
    })


@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return '', 204
    return jsonify({
        "status": "ok",
        "message": "Backend ONLINE",
        "spacy": True,
        "pymusas": HAS_PYMUSAS,
        "pymupdf": HAS_PYMUPDF,
        "pipes": nlp.pipe_names,
        "host": request.host
    })


@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_text():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text']
        doc = nlp(text)
        tokens = []

        for token in doc:
            if HAS_PYMUSAS and hasattr(token._, 'pymusas_tags'):
                semantic = token._.pymusas_tags[0] if token._.pymusas_tags else 'Z99'
            else:
                semantic = get_semantic_fallback(token)

            tokens.append({
                "word": token.text,
                "pos": token.pos_,
                "tag": token.tag_,
                "semantic": semantic,
                "dep": token.dep_,
                "head": token.head.i,
                "lemma": token.lemma_,
                "is_stop": token.is_stop,
                "is_punct": token.is_punct
            })

        return jsonify({
            "tokens": tokens,
            "num_tokens": len(tokens),
            "corpus_name": data.get('corpus_name', 'unnamed'),
            "has_pymusas": HAS_PYMUSAS
        })

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/highlight_pdf', methods=['POST', 'OPTIONS'])
def highlight_pdf():
    if request.method == 'OPTIONS':
        return '', 204
    
    if not HAS_PYMUPDF:
        return jsonify({"error": "PyMuPDF not installed"}), 400

    try:
        import fitz
        import json
        
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        pdf_file = request.files['file']
        phrases = json.loads(request.form.get('phrases', '[]'))
        highlight_color = request.form.get('color', '#FFFF00')

        if not phrases:
            return jsonify({"error": "No phrases provided"}), 400

        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))

        color_rgb = hex_to_rgb(highlight_color)
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        highlight_count = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            for phrase in phrases:
                for inst in page.search_for(phrase, flags=fitz.TEXT_PRESERVE_WHITESPACE):
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=color_rgb)
                    highlight.update()
                    highlight_count += 1

        output_buffer = io.BytesIO()
        doc.save(output_buffer)
        doc.close()
        output_buffer.seek(0)

        original_name = pdf_file.filename.rsplit('.', 1)[0]
        output_filename = f"{original_name}_highlighted.pdf"

        response = make_response(output_buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{output_filename}"'

        print(f"✓ Highlighted {highlight_count} occurrences")
        return response

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def get_semantic_fallback(token):
    pos = token.pos_
    lemma = token.lemma_.lower()

    emotion_pos = {'happy', 'joy', 'delighted', 'pleased', 'excited', 'love', 'wonderful'}
    emotion_neg = {'sad', 'angry', 'fear', 'hate', 'anxious', 'worried', 'upset'}
    if lemma in emotion_pos: return 'E1.1+'
    if lemma in emotion_neg: return 'E1.1-'

    movement = {'go', 'come', 'move', 'walk', 'run', 'travel', 'arrive', 'leave'}
    if lemma in movement: return 'M1'

    speech = {'say', 'tell', 'speak', 'talk', 'communicate', 'discuss', 'ask'}
    if lemma in speech: return 'Q2.2'

    thought = {'think', 'believe', 'know', 'understand', 'consider', 'realize'}
    if lemma in thought: return 'X2.1'

    if pos == 'NOUN': return 'O2'
    elif pos == 'PROPN': return 'Z3'
    elif pos == 'VERB': return 'A3+'
    elif pos == 'ADJ': return 'A5'
    elif pos == 'ADV': return 'A13'
    elif pos == 'NUM': return 'N1'
    elif pos in ['ADP', 'DET']: return 'Z5'
    elif pos == 'PRON': return 'Z8'
    return 'Z99'


if __name__ == '__main__':
    print("\n" + "="*60)
    print("CorBas Backend - NO CACHE VERSION")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=True)

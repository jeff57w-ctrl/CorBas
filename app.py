"""
CorBas Backend - Full Version for Render.com
University of Basra - Linguistics Department
All features included: Real PyMUSAS + spaCy
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import spacy
import os

app = Flask(__name__, static_folder='.')
CORS(app)

# Load spaCy model
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# Load PyMUSAS
print("Loading PyMUSAS...")
HAS_PYMUSAS = False
try:
    from pymusas.rankers.lexicon_entry import ContextualRuleBasedRanker
    from pymusas.taggers.rules.single_word import SingleWordRule
    from pymusas.taggers.rules.mwe import MWERule
    from pymusas.pos_mapper import UPOS_TO_USAS_CORE
    import pymusas.lexicon_collection
    
    lexicon_lookup = pymusas.lexicon_collection.LexiconCollection.from_tsv(
        tsv_file_path=None,
        include_pos=True
    )
    
    ranker = ContextualRuleBasedRanker(*pymusas.rankers.lexicon_entry.ContextualRuleBasedRanker.get_construction_arguments(lexicon_lookup))
    single_word_rule = SingleWordRule(lexicon_lookup, ranker)
    mwe_rule = MWERule(lexicon_lookup, ranker)
    
    config = {
        "rules": [mwe_rule, single_word_rule],
        "ranker": ranker,
        "pos_mapper": UPOS_TO_USAS_CORE
    }
    
    nlp.add_pipe('pymusas_rule_based_tagger', config=config, last=True)
    HAS_PYMUSAS = True
    print("✓ PyMUSAS loaded successfully!")
except Exception as e:
    print(f"⚠ PyMUSAS loading failed: {e}")
    print("Using fallback semantic tagger")
    HAS_PYMUSAS = False

print("Backend ready!")
print(f"Pipeline: {nlp.pipe_names}")


@app.route('/')
def index():
    """Serve the main HTML file"""
    return send_from_directory('.', 'index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "message": "CorBas backend is running on Render.com",
        "spacy": True,
        "pymusas": HAS_PYMUSAS,
        "pipes": nlp.pipe_names
    })


@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Analyze text with spaCy and PyMUSAS"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text']
        corpus_name = data.get('corpus_name', 'unnamed')
        
        print(f"Analyzing: {corpus_name} ({len(text)} chars)")
        
        # Process with spaCy + PyMUSAS
        doc = nlp(text)
        
        tokens = []
        for token in doc:
            # Get PyMUSAS semantic tags
            if HAS_PYMUSAS and hasattr(token._, 'pymusas_tags'):
                semantic_tags = token._.pymusas_tags
                primary_semantic = semantic_tags[0] if semantic_tags else 'Z99'
            else:
                primary_semantic = get_semantic_fallback(token)
            
            tokens.append({
                "word": token.text,
                "pos": token.pos_,
                "tag": token.tag_,
                "semantic": primary_semantic,
                "dep": token.dep_,
                "head": token.head.i,
                "lemma": token.lemma_,
                "is_stop": token.is_stop,
                "is_punct": token.is_punct
            })
        
        print(f"✓ Analyzed {len(tokens)} tokens")
        
        return jsonify({
            "tokens": tokens,
            "num_tokens": len(tokens),
            "corpus_name": corpus_name,
            "has_pymusas": HAS_PYMUSAS
        })
    
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def get_semantic_fallback(token):
    """Fallback semantic tagger"""
    pos = token.pos_
    lemma = token.lemma_.lower()
    
    emotion_positive = {'happy', 'joy', 'delighted', 'pleased', 'excited', 'love', 'wonderful'}
    emotion_negative = {'sad', 'angry', 'fear', 'hate', 'anxious', 'worried', 'upset', 'depressed'}
    
    if lemma in emotion_positive:
        return 'E1.1+'
    if lemma in emotion_negative:
        return 'E1.1-'
    
    movement_verbs = {'go', 'come', 'move', 'walk', 'run', 'travel', 'arrive', 'leave', 'enter'}
    if lemma in movement_verbs:
        return 'M1'
    
    speech_verbs = {'say', 'tell', 'speak', 'talk', 'communicate', 'discuss', 'mention', 'ask'}
    if lemma in speech_verbs:
        return 'Q2.2'
    
    thought_verbs = {'think', 'believe', 'know', 'understand', 'consider', 'realize', 'remember'}
    if lemma in thought_verbs:
        return 'X2.1'
    
    positive_adj = {'good', 'great', 'excellent', 'wonderful', 'amazing', 'beautiful', 'perfect', 'nice'}
    negative_adj = {'bad', 'poor', 'terrible', 'awful', 'horrible', 'ugly', 'wrong', 'worse'}
    
    if lemma in positive_adj:
        return 'A5.1+'
    if lemma in negative_adj:
        return 'A5.1-'
    
    time_words = {'today', 'tomorrow', 'yesterday', 'now', 'then', 'soon', 'later', 'before', 'after'}
    if lemma in time_words:
        return 'T1'
    
    if pos == 'NOUN':
        return 'O2'
    elif pos == 'PROPN':
        return 'Z3'
    elif pos == 'VERB':
        return 'A3+'
    elif pos == 'ADJ':
        return 'A5'
    elif pos == 'ADV':
        return 'A13'
    elif pos == 'NUM':
        return 'N1'
    elif pos == 'ADP':
        return 'Z5'
    elif pos == 'DET':
        return 'Z5'
    elif pos == 'PRON':
        return 'Z8'
    else:
        return 'Z99'


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"\n{'='*60}")
    print("CorBas Backend - Render.com Deployment")
    print(f"{'='*60}")
    print(f"Running on port {port}")
    print(f"PyMUSAS: {'✓ Enabled' if HAS_PYMUSAS else '✗ Fallback mode'}")
    print(f"{'='*60}\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)

import os
import tempfile
import whisper
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import base64
import json

app = Flask(__name__, template_folder='.', static_folder='.')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Load Whisper model
print("🔄 Loading Whisper model for Urdu...")
model = whisper.load_model("medium")
print("✅ Model loaded successfully!")

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'ogg', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        # Check for file upload
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save temporary file
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'webm'
        temp_path = os.path.join(tempfile.gettempdir(), f"audio_{os.urandom(8).hex()}.{ext}")
        file.save(temp_path)
        
        try:
            # Transcribe with Urdu language
            result = model.transcribe(
                temp_path,
                language="ur",
                fp16=False,
                task="transcribe"
            )
            text = result["text"].strip()
            
            if not text:
                text = "معذرت، کوئی آواز نہیں پہچانی گئی"
            
            return jsonify({
                'success': True,
                'text': text,
                'language': 'urdu'
            })
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'Transcription failed: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model': 'whisper-medium'})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🎤 UrduVoice AI is starting...")
    print("📍 Open http://localhost:5000 in your browser")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
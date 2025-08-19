import os
import subprocess
import json

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# -------------------- Flask App Configuration --------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
# -------------------- Home Page --------------------
@app.route('/')
def index():
    """Renders the main homepage."""
    return render_template('index.html')

# -------------------- Extract Emotion & Transcription --------------------
def process_output(output):
    """Parses the output from real_time_ser.py to extract emotion and transcription."""
    emotion, transcription = "Unknown", "No transcription"

    if not output:
        print("❌ No output received from real_time_ser.py")
        return emotion, transcription

    print("📝 Full Output from Model:", output)  # ✅ Debugging Step

    if not output:
        print("❌ No output received from real_time_ser.py")
        return "Unknown", "No transcription"

    last_line = output.strip().split("\n")[-1] if output else ""

    if "EMOTION:" in last_line and "TRANSCRIPTION:" in last_line:
        parts = last_line.split("|")
        for part in parts:
            if "EMOTION:" in part:
                emotion = part.replace("EMOTION:", "").strip()
            elif "TRANSCRIPTION:" in part:
                transcription = part.replace("TRANSCRIPTION:", "").strip()

    return emotion, transcription

# -------------------- Upload and Process Audio File --------------------
@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    file.save(file_path)

    try:
        # ✅ Run with absolute path & force UTF-8 decoding
        result = subprocess.run(
            ['python', 'real_time_ser.py', file_path], 
            capture_output=True, 
            text=True, 
            encoding='utf-8'
        )

        print("\n📝 Raw Output from real_time_ser.py:\n", result.stdout)  # ✅ Debugging

        if result.returncode != 0:
            print(f"❌ Error running real_time_ser.py: {result.stderr}")  # ✅ Debugging
            return jsonify({"error": "Failed to process audio file"}), 500

        # ✅ Extract the **last line** (which contains JSON)
        json_line = result.stdout.strip().split("\n")[-1]
        json_output = json.loads(json_line)

        return jsonify(json_output)  # ✅ Correctly pass JSON to frontend

    except json.JSONDecodeError:
        print("❌ JSON Decode Error: Output is not valid JSON!")  # ✅ Debugging
        return jsonify({"error": "Invalid JSON output"}), 500
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500
# -------------------- Handle Live Audio Recording --------------------
@app.route('/record', methods=['POST'])
def record_audio():
    """Handles live audio recording from frontend and predicts emotion & transcription."""
    audio_data = request.files.get('audio')
    if not audio_data:
        return jsonify({'error': 'No audio received'}), 400

    filename = "live_recording.wav"
    file_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))  # ✅ Absolute path

    # Save the received audio file
    # Save the received audio file correctly in WAV format
    audio_data.save(file_path)

    # Convert audio to proper WAV format using pydub (if needed)
    from pydub import AudioSegment
    audio = AudioSegment.from_file(file_path)
    audio.export(file_path, format="wav")


    print(f"📁 File saved: {file_path}, Size: {os.path.getsize(file_path)} bytes")  # ✅ Debugging

    # Run `real_time_ser.py` for emotion & transcription
    try:
        result = subprocess.run(
            ['python', 'real_time_ser.py', file_path], 
            capture_output=True, 
            text=True, 
            encoding='utf-8'  # ✅ Force UTF-8 decoding
        )

        if result.returncode != 0:
            print(f"❌ Error running real_time_ser.py: {result.stderr}")  # ✅ Debugging
            return jsonify({"error": "Error running speech emotion recognition"}), 500

        # ✅ Extract the **last line** (which contains JSON)
        json_line = result.stdout.strip().split("\n")[-1]
        json_output = json.loads(json_line)  # ✅ Parse JSON correctly

        return jsonify(json_output)  # ✅ Return JSON to frontend

    except json.JSONDecodeError:
        print("❌ JSON Decode Error: Output is not valid JSON!")  # ✅ Debugging
        return jsonify({"error": "Invalid JSON output"}), 500
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


# -------------------- Run Flask Server --------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

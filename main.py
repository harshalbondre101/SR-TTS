from flask import Flask, request, render_template, jsonify
import speech_recognition as sr
from pydub import AudioSegment
import io
import os
import replicate

app = Flask(__name__)
recognizer = sr.Recognizer()

# Set the API token in the environment
os.environ["REPLICATE_API_TOKEN"] = "r8_1D9C9cyT8YnQN67nOsUJq1GiujSqIqo4g6irG"

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/recognize', methods=['POST'])
def recognize_speech():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    try:
        # Convert the uploaded audio file to a PCM WAV format
        audio = AudioSegment.from_file(io.BytesIO(file.read()))
        audio = audio.set_channels(1).set_frame_rate(16000)
        wav_io = io.BytesIO()
        audio.export(wav_io, format='wav')
        wav_io.seek(0)

        # Recognize speech using Google's speech recognition
        with sr.AudioFile(wav_io) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return jsonify({"text": text})
    except sr.RequestError:
        return jsonify({"error": "API was unreachable or unresponsive"})
    except sr.UnknownValueError:
        return jsonify({"error": "Unable to recognize speech"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/generate-audio', methods=['POST'])
def generate_audio():
    data = request.json
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        output = replicate.run(
            "suno-ai/bark:b76242b40d67c76ab6742e987628a2a9ac019e11d56ab96c4e91ce03b79b2787",
            input={
                "prompt": prompt,
                "text_temp": 0.7,
                "output_full": False,
                "waveform_temp": 0.7,
                "history_prompt": "announcer"
            }
        )
        audio_out = output.get('audio_out', '')
        
        if not audio_out:
            return jsonify({"error": "Audio generation failed"}), 500
        
        return jsonify({"audio_out": audio_out})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

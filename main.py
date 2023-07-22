from flask import Flask, render_template, request
from transformers import pipeline
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

app = Flask(__name__)

# Load the translation models
en_hi_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
hi_en_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")

# Load the summarization model
summarizer = pipeline("summarization")

# Load ASR model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text_input = request.form["text_input"]
        translation_direction = request.form["translation_direction"]
        action = request.form["action"]  # New parameter to determine the action

        if action == "translate":
            # Check if audio_input is in request.files (i.e., if user uploaded audio)
            if "audio_input" in request.files:
                audio_file = request.files["audio_input"]
                if audio_file.filename != "":
                    # Process audio input and perform translation
                    speech, rate = librosa.load(audio_file, sr=16000)
                    input_values = tokenizer(speech, return_tensors='pt').input_values
                    # Store logits (non-normalized predictions)
                    logits = model(input_values).logits
                    # Store predicted id's
                    predicted_ids = torch.argmax(logits, dim=-1)
                    # Decode the audio to generate text
                    text_input = tokenizer.decode(predicted_ids[0])

            if translation_direction == "en-hi":
                translated_text = en_hi_translator(text_input)[0]["translation_text"]
            elif translation_direction == "hi-en":
                translated_text = hi_en_translator(text_input)[0]["translation_text"]
            else:
                translated_text = "Invalid translation direction"

            return render_template("index1.html", translated_text=translated_text, text_input=text_input,
                                   translation_direction=translation_direction)

        elif action == "summarize":
            # Perform text summarization
            summarized_text = summarizer(text_input)[0]["summary_text"]
            return render_template("index1.html", summarized_text=summarized_text, text_input=text_input,
                                   translation_direction=translation_direction)

    return render_template("index1.html", translation_direction="en-hi", text_input="")


if __name__ == "__main__":
    app.run(debug=True, port=5000)

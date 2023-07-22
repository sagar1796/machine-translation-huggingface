from IPython.display import Audio, display
from transformers import pipeline

import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
#load model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Replace 'audio.mp3' with the actual path to your audio file
audio_file = "static/my_voice_0_0.wav"

# Replace 'sample_rate' with the actual sample rate of your audio file
# sample_rate = 22050

# print(display(Audio(audio_file, rate=sample_rate, autoplay=True)))

#load any audio file of your choice
speech, rate = librosa.load( audio_file,sr=16000)
input_values = tokenizer(speech, return_tensors = 'pt').input_values
#Store logits (non-normalized predictions)
logits = model(input_values).logits
#Store predicted id's
predicted_ids = torch.argmax(logits, dim =-1)
#decode the audio to generate text
transcriptions = tokenizer.decode(predicted_ids[0])
print(transcriptions)



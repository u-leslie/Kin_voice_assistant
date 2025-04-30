from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
from torchaudio.transforms import Resample
import os
import soundfile as sf
from huggingface_hub import login
from dotenv import load_dotenv
from transformers import VitsModel, VitsTokenizer




# Load environment variables (for Hugging Face token)
load_dotenv()
api_token = os.getenv('HUG_TOKEN')
login(api_token)

# Optional: ensure ffmpeg is on PATH (for audio playback on Windows)
os.environ["PATH"] += os.pathsep + r"C:\Users\HP PAVILLION\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"

# Load Whisper model for Kinyarwanda transcription
asr_model = WhisperForConditionalGeneration.from_pretrained("mbazaNLP/Whisper-Small-Kinyarwanda")
asr_processor = WhisperProcessor.from_pretrained("mbazaNLP/Whisper-Small-Kinyarwanda")

# Load VITS model and tokenizer
tts_model = VitsModel.from_pretrained("facebook/mms-tts-kin")
tts_tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-kin")

# Define question-answer pairs
qa_pairs = {
    "amakuru": "Ni meza, urakoze!",
    "witwa nde": "Nitwa Umuhire",
    "ubuzima bumeze gute": "Bumeze neza!",
    "ikinyarwanda ni iki": "Ni ururimi kavukire rw'Abanyarwanda.",
    "amakuru yawe": "Ni meza, ndashimira!"
}

# Output directory
output_folder = 'outputs/'
os.makedirs(output_folder, exist_ok=True)

def speak_answer(answer_text, output_file):
    inputs = tts_tokenizer(answer_text, return_tensors="pt")

    with torch.no_grad():
        speech = tts_model(**inputs).waveform

    # Save as .wav
    torchaudio.save(output_file, speech, sample_rate=tts_model.config.sampling_rate)

    # Play audio (Windows)
    os.system(f'start {output_file}')


# Folder containing .wav input files
audio_folder = 'audio/'

# Process each audio file
for file_name in os.listdir(audio_folder):
    if file_name.endswith('.wav'):
        file_path = os.path.join(audio_folder, file_name)

        # Load and resample if needed
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != 16000:
            resampler = Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Transcribe audio
        inputs = asr_processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
        predicted_ids = asr_model.generate(
            inputs["input_features"],
            max_new_tokens=20,
            no_repeat_ngram_size=1,
            suppress_tokens=[],
        )
        transcription = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        recognized_text = transcription.lower().strip()
        print(f"Recognized Text: {recognized_text}")

        # Match and respond
        matched_answer = next((a for q, a in qa_pairs.items() if q in recognized_text), None)

        if matched_answer:
            print(f"Answer: {matched_answer}")
            output_path = os.path.join(output_folder, f"{file_name.replace('.wav', '')}_answer.wav")
            speak_answer(matched_answer, output_path)
        else:
            print("No matching answer found.")

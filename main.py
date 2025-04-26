# import whisper
# from gtts import gTTS
# import os

# # Tell pydub and whisper where ffmpeg is manually
# os.environ["PATH"] += os.pathsep + r"C:\Users\HP PAVILLION\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"


# # Load model
# model = whisper.load_model("small")

# # Question-Answer pairs
# qa_pairs = {
#     "amakuru": "Ni meza, urakoze!",
#     "witwa nde": "Nitwa Umufasha w'Ijwi.",
#     "ubuzima bumeze gute": "Bumeze neza!",
#     "ikinyarwanda ni iki": "Ni ururimi kavukire rw'Abanyarwanda.",
#     "amakuru yawe": "Ni meza, ndashimira!"
# }

# # Create outputs folder if not exist
# output_folder = 'outputs/'
# os.makedirs(output_folder, exist_ok=True)

# # Helper function to generate and play answer
# def speak_answer(answer_text, output_file):
#     tts = gTTS(text=answer_text, lang='rw')
#     tts.save(output_file)
#     os.system(f"start {output_file}")  # Windows
#     # os.system(f"afplay {output_file}")  # Mac (if you're using Mac)

# # Directory of audio files
# audio_folder = 'audio/'

# # Process each audio
# for file_name in os.listdir(audio_folder):
#     if file_name.endswith('.wav'):
#         file_path = os.path.join(audio_folder, file_name)
#         result = model.transcribe(file_path)  
#         recognized_text = result['text'].lower().strip()
#         print(f"Recognized Text: {recognized_text}")
        
#         # Simple matching
#         matched_answer = None
#         for question, answer in qa_pairs.items():
#             if question in recognized_text:
#                 matched_answer = answer
#                 break
        
#         if matched_answer:
#             print(f"Answer: {matched_answer}")
#             output_file = os.path.join(output_folder, f"{file_name.replace('.wav', '')}_answer.mp3")
#             speak_answer(matched_answer, output_file)
#         else:
#             print("No matching answer found.")



from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
from gtts import gTTS
import os

# Tell pydub and ffmpeg where ffmpeg is manually (Important for Windows)
os.environ["PATH"] += os.pathsep + r"C:\Users\HP PAVILLION\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"

# Load fine-tuned KinyaWhisper model and processor from Hugging Face
model = WhisperForConditionalGeneration.from_pretrained("benax-rw/KinyaWhisper")
processor = WhisperProcessor.from_pretrained("benax-rw/KinyaWhisper")

# Question-Answer pairs
qa_pairs = {
    "amakuru": "Ni meza, urakoze!",
    "witwa nde": "Nitwa Umufasha w'Ijwi.",
    "ubuzima bumeze gute": "Bumeze neza!",
    "ikinyarwanda ni iki": "Ni ururimi kavukire rw'Abanyarwanda.",
    "amakuru yawe": "Ni meza, ndashimira!"
}

# Create outputs folder if not exist
output_folder = 'outputs/'
os.makedirs(output_folder, exist_ok=True)

# Helper function to generate and play answer
def speak_answer(answer_text, output_file):
    tts = gTTS(text=answer_text, lang='rw')
    tts.save(output_file)
    os.system(f"start {output_file}")  # Windows
    # os.system(f"afplay {output_file}")  # Mac (if you're using Mac)

# Directory of audio files
audio_folder = 'audio/'

# Process each audio
for file_name in os.listdir(audio_folder):
    if file_name.endswith('.wav'):
        file_path = os.path.join(audio_folder, file_name)

        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(file_path)
        inputs = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt")

        # Generate prediction
        predicted_ids = model.generate(inputs["input_features"])
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        recognized_text = transcription.lower().strip()
        print(f"Recognized Text: {recognized_text}")

        # Simple matching
        matched_answer = None
        for question, answer in qa_pairs.items():
            if question in recognized_text:
                matched_answer = answer
                break

        if matched_answer:
            print(f"Answer: {matched_answer}")
            output_file = os.path.join(output_folder, f"{file_name.replace('.wav', '')}_answer.mp3")
            speak_answer(matched_answer, output_file)
        else:
            print("No matching answer found.")

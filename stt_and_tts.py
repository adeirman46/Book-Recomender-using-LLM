import sounddevice as sd
import numpy as np
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from kokoro import KPipeline
from IPython.display import Audio

def record_audio(duration, sample_rate=16000):
    """Record audio from microphone"""
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate),
                  samplerate=sample_rate,
                  channels=1,
                  dtype=np.float32)
    sd.wait()
    print("Recording finished!")
    return audio.flatten()

def transcribe_audio(audio_array, sample_rate=16000):
    """Transcribe audio using Whisper"""
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.config.forced_decoder_ids = None
    
    # Process audio with explicit language setting and attention mask
    inputs = processor(
        audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt",
        return_attention_mask=True
    )
    
    # Generate with explicit settings
    predicted_ids = model.generate(
        inputs.input_features,
        attention_mask=inputs.attention_mask,
        language="en",
        task="transcribe"
    )
    
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

def speak_text(text):
    """Convert text to speech using Kokoro and play it immediately"""
    # Initialize Kokoro pipeline
    pipeline = KPipeline(lang_code='a')  # American English
    
    # Generate audio
    generator = pipeline(
        text,
        voice='af_heart',
        speed=1,
        split_pattern=r'\n+'
    )
    
    # Get the first (and only) audio segment and play it immediately
    for _, _, audio in generator:
        # Play audio directly using sounddevice
        sd.play(audio, samplerate=24000)
        sd.wait()  # Wait until audio is finished playing

def main():
    try:
        # Record audio (5 seconds)
        audio_array = record_audio(5)
        
        # Transcribe the audio
        print("Transcribing...")
        transcribed_text = transcribe_audio(audio_array)
        print("\nTranscription:", transcribed_text)
        
        # Convert transcription to speech and play it automatically
        print("\nGenerating and playing speech...")
        speak_text(transcribed_text)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
import sounddevice as sd
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from kokoro import KPipeline
from langchain_community.llms import Ollama

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
    
    inputs = processor(
        audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt",
        return_attention_mask=True
    )
    
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
    pipeline = KPipeline(lang_code='a')
    generator = pipeline(
        text,
        voice='af_heart',
        speed=1,
        split_pattern=r'\n+'
    )
    
    for _, _, audio in generator:
        sd.play(audio, samplerate=24000)
        sd.wait()

def get_llm_response(text):
    """Get driving commands interpretation from LLM"""
    llm = Ollama(model="deepseek-r1:1.5b")
    
    system_prompt = """You are a JSON-only driving command interpreter. ONLY OUTPUT VALID JSON - NO OTHER TEXT ALLOWED.

    Valid fields (include only relevant ones):
    - Throttle (0-100)
    - Brake (0-100)
    - Steer (-45 to 45, negative for left, positive for right)
    - LeftSignal (0 or 1)
    - RightSignal (0 or 1)
    DON'T ADD ANY OTHER FIELDS. JUST CORRELATE THE COMMANDS TO THE FIELDS.
    BECAUSE MAYBE THE INPUT TEXT CONTAINS TYPOS

    Input: "drive slowly" 
    {"Throttle": 20}

    Input: "brake immediately"
    {"Brake": 100}

    Input: "10 degrees left"
    {"Steer": -10}

    Input: "turn on left signal"
    {"LeftSignal": 1}

    REMEMBER: OUTPUT ONLY JSON, NO THINKING, NO EXPLANATIONS!
    JUST OUTPUT ONE COMMAND PER INPUT. I MEAN {"COMMAND": VALUE} ONLY.
    COMMAND IS JUST THROTTLE, BRAKE, STEER, LEFTSIGNAL, OR RIGHTSIGNAL.
    """
    
    prompt = f"Convert to JSON ONLY: '{text}'"
    return llm.invoke(system_prompt + "\n" + prompt)

def get_clean_response(response):
    """Get response after </think> tag"""
    if '</think>' in response:
        return response.split('</think>')[-1].strip()
    return response

def main():
    try:
        # Record audio (5 seconds)
        audio_array = record_audio(5)
        
        # Transcribe the audio
        print("Transcribing...")
        transcribed_text = transcribe_audio(audio_array)
        print("\nTranscription:", transcribed_text)
        
        # Get LLM response
        print("\nGetting LLM response...")
        full_response = get_llm_response(transcribed_text)
        print("Full response:", full_response)
        
        # Clean response and speak only the JSON part
        clean_response = get_clean_response(full_response)
        print("\nSpeaking clean response:", clean_response)
        speak_text(clean_response)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
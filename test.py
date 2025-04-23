import pyaudio
import wave
import requests
import time
import os

# AssemblyAI API configuration
API_KEY = "f6b537a6b4f140dfbead28720751b78e"
HEADERS = {
    "authorization": API_KEY,
    "content-type": "application/json"
}
API_ENDPOINT = "https://api.assemblyai.com/v2/transcript"

def record_audio(filename="recording.wav", seconds=5):
    """Record audio from microphone for specified number of seconds"""
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 44100
    
    p = pyaudio.PyAudio()
    
    print(f"Recording for {seconds} seconds...")
    
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)
    
    frames = []
    
    # Record for the specified duration
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    print("Recording complete.")
    
    # Save the recording to a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return filename

def transcribe_audio(audio_file):
    """Send audio file to AssemblyAI for transcription"""
    print("Uploading audio file...")
    
    # Upload the audio file to AssemblyAI
    upload_endpoint = "https://api.assemblyai.com/v2/upload"
    with open(audio_file, "rb") as f:
        response = requests.post(upload_endpoint, headers=HEADERS, data=f)
    
    audio_url = response.json()["upload_url"]
    print(f"Audio file uploaded.")
    
    # Request transcription
    transcript_request = {
        "audio_url": audio_url,
        "language_code": "en"  # Change if needed
    }
    
    response = requests.post(API_ENDPOINT, json=transcript_request, headers=HEADERS)
    transcript_id = response.json()["id"]
    
    # Poll for transcription completion
    polling_endpoint = f"{API_ENDPOINT}/{transcript_id}"
    
    print("Waiting for transcription to complete...")
    while True:
        response = requests.get(polling_endpoint, headers=HEADERS)
        status = response.json()["status"]
        
        if status == "completed":
            return response.json()["text"]
        elif status == "error":
            return "Transcription error occurred."
        
        time.sleep(1)

def main():
    """Main function to handle the speech-to-text flow"""
    try:
        print("Automatic speech recognition started")
        print("Press Ctrl+C at any time to exit")
        
        while True:
            print("\nStarting new recording in 3 seconds...")
            time.sleep(3)
            
            # Record audio from microphone
            record_duration = 10  # Recording duration in seconds
            audio_file = record_audio(seconds=record_duration)
            
            # Transcribe the recorded audio
            transcription = transcribe_audio(audio_file)
            
            # Display the transcription result
            print("\nTranscription:")
            print(transcription)
            print("\n" + "-"*50)
            
    except KeyboardInterrupt:
        print("\nExiting the program.")
    finally:
        # Clean up temporary files
        if os.path.exists("recording.wav"):
            os.remove("recording.wav")

if __name__ == "__main__":
    main()
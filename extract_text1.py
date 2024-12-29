"""  
    python extract_text1.py https://youtu.be/video1 https://youtu.be/video2
    The script will process each URL in sequence and print the transcription (or error) for each one.
"""


REQUIRED_PACKAGES = ['yt-dlp', 'openai-whisper']

"""  
You'll also need FFmpeg installed on your system
UBUNTU: sudo apt install ffmpeg
WINDOWS: choco install ffmpeg
"""

try:
    import whisper
    import yt_dlp
    import argparse
except ImportError as e:
    raise ImportError(f"Please install required packages: {str(e)}\n"
                     f"pip install {' '.join(REQUIRED_PACKAGES)}")


def download_youtube_video_audio(url="youtube video url"):
    """
    Downloads YouTube video and returns the file path.
    Handles video download with basic error checking.
    """
    if not url:
        raise ValueError("URL cannot be empty")

    # Download configuration
    ydl_opts = {
        'format': 'bestaudio',
        'outtmpl': 'downloads/audio.%(ext)s',
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192'
        }]
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
            # Return the direct path to the MP3 file
            return 'downloads/audio.mp3'
    except Exception as e:
        raise Exception(f"Download failed: {str(e)}")

def transcribe_audio(audio_path):
    """
    Transcribes audio file to text using OpenAI's Whisper large model.
    Returns the transcribed text with highest accuracy.
    """
    try:
        # Load the large model with CPU device specification
        model = whisper.load_model("large", device="cpu")
        
        # Transcribe the audio
        result = model.transcribe(audio_path)
        
        return result["text"]
    except Exception as e:
        raise Exception(f"Transcription failed: {str(e)}")

def process_youtube_url(url):
    """
    Downloads YouTube video audio and transcribes it to text.
    Returns the transcribed text.
    """
    try:
        # Download the audio
        audio_path = download_youtube_video_audio(url)
        
        # Transcribe the audio to text
        text = transcribe_audio(audio_path)
        
        return text
    except Exception as e:
        raise Exception(f"Processing failed: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download audio and transcribe YouTube videos')
    parser.add_argument('urls', nargs='+', help='One or more YouTube URLs to process')
    
    args = parser.parse_args()
    
    for url in args.urls:
        print(f"\nProcessing: {url}")
        try:
            text = process_youtube_url(url)
            print("Transcription:")
            print(text)
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")


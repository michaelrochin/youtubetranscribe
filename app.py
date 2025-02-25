import streamlit as st
import yt_dlp
import whisper
import os

def download_audio(url, cookie_file, output_dir, progress_bar, status_placeholder):
    # Configure yt-dlp to download the best audio and convert to MP3 using ffmpeg.
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
    }

    # Handle optional cookies file.
    cookie_path = None
    if cookie_file is not None:
        cookie_path = os.path.join(output_dir, "cookies_temp.txt")
        with open(cookie_path, 'wb') as f:
            f.write(cookie_file.getvalue())
        ydl_opts['cookiefile'] = cookie_path

    # Progress hook for updating the UI.
    def progress_hook(d):
        if d['status'] == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            downloaded = d.get('downloaded_bytes', 0)
            if total:
                percentage = int(downloaded / total * 100)
                progress_bar.progress(min(percentage, 100))
                status_placeholder.info(f"Downloading: {percentage}%")
        elif d['status'] == 'finished':
            progress_bar.progress(100)
            status_placeholder.success("Download complete!")

    ydl_opts['progress_hooks'] = [progress_hook]

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get info to determine output filename.
            info = ydl.extract_info(url, download=False)
            filename = ydl.prepare_filename(info)
            mp3_file = os.path.splitext(filename)[0] + '.mp3'
            status_placeholder.info("Downloading audio...")
            ydl.download([url])
    except Exception as e:
        st.error(f"Error downloading audio: {e}")
        mp3_file = None

    # Clean up the temporary cookies file.
    if cookie_path and os.path.exists(cookie_path):
        os.remove(cookie_path)

    return mp3_file

def transcribe_audio(mp3_file, model_choice, include_timestamps, status_placeholder):
    # Force device to CPU to avoid GPU-related issues on Streamlit Cloud.
    device = "cpu"
    status_placeholder.info("Loading Whisper model...")
    model = whisper.load_model(model_choice, device=device)
    status_placeholder.info("Transcribing audio...")
    result = model.transcribe(mp3_file)

    # Format the transcript with optional timestamps.
    transcript_lines = []
    def format_timestamp(seconds):
        mm = int(seconds // 60)
        ss = int(seconds % 60)
        return f"[{mm:02d}:{ss:02d}]"
    
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if include_timestamps:
            transcript_lines.append(f"{format_timestamp(seg.get('start', 0))} {text}")
        else:
            transcript_lines.append(text)
    
    transcript_text = "\n".join(transcript_lines)
    return transcript_text

def main():
    st.title("YouTube & Facebook Audio Downloader & Transcriber")

    # ----------- USER INPUTS -----------
    url = st.text_input("Enter video URL (YouTube or Facebook):")
    cookie_file = st.file_uploader("Upload Cookies File (optional)", type=['txt'])
    output_dir = "downloads"
    os.makedirs(output_dir, exist_ok=True)

    model_options = ["tiny", "base", "small", "medium", "large"]
    model_choice = st.selectbox("Select Whisper Model:", model_options, index=1)
    save_audio = st.checkbox("Save audio file (MP3)", value=True)
    do_transcribe = st.checkbox("Transcribe audio", value=True)
    include_timestamps = st.checkbox("Include timestamps in transcript", value=False)

    # ----------- PROCESSING -----------
    if st.button("Download and Process"):
        if not url:
            st.error("Please provide a video URL.")
            return

        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        mp3_file = download_audio(url, cookie_file, output_dir, progress_bar, status_placeholder)
        if not mp3_file:
            st.error("Audio download failed.")
            return

        transcript_text = ""
        if do_transcribe:
            try:
                transcript_text = transcribe_audio(mp3_file, model_choice, include_timestamps, status_placeholder)
            except Exception as e:
                st.error(f"Error during transcription: {e}")
                return

        if not save_audio and os.path.exists(mp3_file):
            os.remove(mp3_file)

        status_placeholder.success("Processing complete!")

        if transcript_text:
            st.subheader("Transcription:")
            st.text_area("Transcribed Text", transcript_text, height=300)
            st.download_button("Download Transcript", transcript_text, "transcript.txt", "text/plain")

if __name__ == "__main__":
    main()

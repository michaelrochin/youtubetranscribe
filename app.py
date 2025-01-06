import streamlit as st
import yt_dlp
import whisper
import os
import torch
import gc

def main():
    st.title("YouTube & Facebook Audio Downloader & Transcriber")

    # ----------- USER INPUTS -----------
    url = st.text_input("Video URL (YouTube or Facebook)")

    # Optional: upload a cookies.txt for age-locked/private videos
    cookie_file = st.file_uploader("Cookies File (optional)", type=['txt'])

    output_dir = "downloads"
    os.makedirs(output_dir, exist_ok=True)

    model_options = ("tiny", "base", "small", "medium", "large")
    model_choice = st.selectbox("Whisper Model:", model_options, index=1)

    save_audio = st.checkbox("Save MP3", value=True)
    do_transcribe = st.checkbox("Transcribe Audio", value=True)
    include_timestamps = st.checkbox("Include Timestamps", value=False)

    # ----------- RUN WHEN BUTTON CLICKED -----------
    if st.button("Download & Process"):
        if not url.strip():
            st.error("Please enter a valid video URL.")
            return

        # 1) Set up yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        }

        # If user provided a cookies file, save it and pass to yt-dlp
        cookie_path = None
        if cookie_file is not None:
            cookie_path = os.path.join(output_dir, "cookies_temp.txt")
            with open(cookie_path, 'w', encoding='utf-8') as f:
                f.write(cookie_file.read().decode('utf-8'))
            ydl_opts['cookiefile'] = cookie_path

        # 2) Download the audio
        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        def download_progress_hook(d):
            if d['status'] == 'downloading':
                total = d.get('total_bytes', 0) or d.get('total_bytes_estimate', 0)
                downloaded = d.get('downloaded_bytes', 0)
                if total > 0:
                    percentage = int((downloaded / total) * 100)
                    progress_bar.progress(min(percentage, 100))
                    status_placeholder.info(f"Downloading: {percentage}%")
            elif d['status'] == 'finished':
                progress_bar.progress(100)
                status_placeholder.success("Download complete!")

        ydl_opts['progress_hooks'] = [download_progress_hook]

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                filename = ydl.prepare_filename(info)
                mp3_file = os.path.splitext(filename)[0] + '.mp3'

                status_placeholder.info("Downloading audio...")
                ydl.download([url])
        except Exception as e:
            st.error(f"Error while downloading: {e}")
            # Clean up any cookies file
            if cookie_path and os.path.exists(cookie_path):
                os.remove(cookie_path)
            return

        # 3) Transcription (if selected)
        transcript_text = ""
        if do_transcribe:
            try:
                status_placeholder.info("Loading Whisper model...")
                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"

                model = whisper.load_model(model_choice).to(device)
                model.eval()

                status_placeholder.info("Transcribing audio...")
                result = model.transcribe(mp3_file)

                # Clean GPU memory if used
                if device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()

                # Format the transcript
                def format_timestamp(sec):
                    mm = int(sec // 60)
                    ss = int(sec % 60)
                    return f"[{mm:02d}:{ss:02d}]"

                lines = []
                for seg in result["segments"]:
                    segment_start = seg["start"]
                    segment_text = seg["text"].strip()

                    if include_timestamps:
                        stamp = format_timestamp(segment_start)
                        lines.append(f"{stamp} {segment_text}")
                    else:
                        lines.append(segment_text)

                transcript_text = "\n".join(lines)

            except Exception as e:
                st.error(f"Error while transcribing: {e}")
                if cookie_path and os.path.exists(cookie_path):
                    os.remove(cookie_path)
                return

        # 4) Optional Cleanup
        if not save_audio and os.path.exists(mp3_file):
            os.remove(mp3_file)

        status_placeholder.success("Processing complete!")

        # Show transcript if available
        if transcript_text:
            st.subheader("Transcription:")
            st.text_area("Transcribed Text", transcript_text, height=300)
            st.download_button(
                "Download Transcript",
                data=transcript_text,
                file_name="transcript.txt",
                mime="text/plain"
            )

        # Remove temp cookie file if created
        if cookie_path and os.path.exists(cookie_path):
            os.remove(cookie_path)

if __name__ == "__main__":
    main()

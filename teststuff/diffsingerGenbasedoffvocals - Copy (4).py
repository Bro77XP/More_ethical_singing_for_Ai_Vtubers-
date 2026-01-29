import whisper
import librosa
import numpy as np
import json
from pathlib import Path
import requests
from bs4 import BeautifulSoup

def fetch_lyrics_from_genius(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        lyrics_div = soup.find('div', class_='lyrics')
        if lyrics_div:
            lyrics = lyrics_div.get_text()
            return lyrics
        return None
    except:
        return None

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True)
    return result

def extract_pitch(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr)
    hop_length = 256
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), hop_length=hop_length, sr=sr)
    return f0, voiced_flag, sr, hop_length

def generate_ust(lyrics_segments, pitch_data, output_path, audio_duration):
    f0, voiced_flag, sr, hop_length = pitch_data
    tempo = 120  # Default tempo
    # Sort segments by start time
    lyrics_segments = sorted(lyrics_segments, key=lambda x: x['start'])
    ticks_per_second = 480 * tempo / 60  # ticks per beat * BPM / 60 seconds

    ust_content = f"""[#VERSION]
UST Version 1.2
[#SETTING]
Tempo={tempo:.2f}
Tracks=1
ProjectName=Generated from Vocals
VoiceDir=%VOICE%\\Default
OutFile=
CacheDir=%CACHE%
Tool1=wavtool.exe
Tool2=resampler.exe
Mode2=True
"""

    # Shift times to start from the first segment
    if lyrics_segments:
        first_start = lyrics_segments[0]['start']
        for segment in lyrics_segments:
            segment['start'] -= first_start
            segment['end'] -= first_start
    else:
        first_start = 0

    # Calculate total segments duration and scale factor to match audio duration
    total_segments_duration = lyrics_segments[-1]['end'] if lyrics_segments else 0
    scale_factor = audio_duration / total_segments_duration if total_segments_duration > 0 else 1
    print(f"Audio duration: {audio_duration:.2f}s, Segments duration: {total_segments_duration:.2f}s, Scale factor: {scale_factor:.2f}")

    # Apply scale factor to segment times to match audio duration
    for segment in lyrics_segments:
        segment['start'] *= scale_factor
        segment['end'] *= scale_factor

    note_index = 0
    previous_end = 0.0

    for segment in lyrics_segments:
        start_time = segment['start']
        end_time = segment['end']
        lyric = segment.get('text', segment.get('word', '')).strip()
        if not lyric:
            continue

        # Insert rest if there's a gap
        gap = start_time - previous_end
        if gap > 0:
            rest_length = int(gap * ticks_per_second)
            if rest_length > 0:
                ust_content += f"[#{note_index:04d}]\n"
                ust_content += f"Length={rest_length}\n"
                ust_content += "Lyric=R\n"
                ust_content += "NoteNum=60\n"
                ust_content += "PreUtterance=\n"
                ust_content += "VoiceOverlap=\n"
                ust_content += "Intensity=100\n"
                ust_content += "Modulation=0\n"
                ust_content += "Velocity=100\n"
                ust_content += "Flags=\n"
                note_index += 1

        # Calculate length for the lyric note
        duration = end_time - start_time
        length = int(duration * ticks_per_second)

        # Calculate average pitch
        start_frame = int((start_time + first_start) * sr / hop_length)
        end_frame = int((end_time + first_start) * sr / hop_length)
        if start_frame < len(f0) and end_frame <= len(f0):
            segment_f0 = f0[start_frame:end_frame]
            valid_f0 = segment_f0[voiced_flag[start_frame:end_frame]]
            if len(valid_f0) > 0:
                avg_f0 = np.mean(valid_f0)
                if avg_f0 > 0:
                    note_num = int(round(69 + 12 * np.log2(avg_f0 / 440)))
                    note_num = max(54, min(70, note_num))  # Constrain to F#3 (54) to A#4 (70)
                else:
                    note_num = 60
            else:
                note_num = 60
        else:
            note_num = 60

        ust_content += f"[#{note_index:04d}]\n"
        ust_content += f"Length={length}\n"
        ust_content += f"Lyric={lyric}\n"
        ust_content += f"NoteNum={note_num}\n"
        ust_content += "PreUtterance=\n"
        ust_content += "VoiceOverlap=\n"
        ust_content += "Velocity=100\n"
        ust_content += "Intensity=100\n"
        ust_content += "Modulation=0\n"
        ust_content += "Flags=g0B0H0P86\n"
        ust_content += "PBS=-40;0\n"
        ust_content += "PBW=80\n"
        ust_content += "PBY=0\n"
        ust_content += "PBM=\n"
        ust_content += "VBR=75,175,25,10,10,0,0\n"
        note_index += 1
        previous_end = end_time

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ust_content)

def main():
    # Set to False to disable Genius lyric fetching and or you u justn wanna mess with this for fun <3
    use_genius = False
    # Set to True to use local lyrics from lyrics.txt instead of transcribed or Genius
    use_local_lyrics = True

    # Look for vocals.wav in the script's directory
    script_dir = Path(__file__).parent
    audio_path = script_dir / "vocals.wav"

    print(f"Current directory: {Path.cwd()}")
    print(f"Looking for: {audio_path}")
    print(f"Exists: {audio_path.exists()}")

    if not audio_path.exists():
        print(f"Audio file {audio_path} not found.")
        return

    correct_words = []
    if use_local_lyrics:
        print("Using local lyrics from lyrics.txt...")
        lyrics_file = script_dir / "lyrics.txt"
        if lyrics_file.exists():
            with open(lyrics_file, 'r', encoding='utf-8') as f:
                lyrics_text = f.read()
            correct_words = [word.strip() for word in lyrics_text.split() if word.strip()]
            print(f"Loaded {len(correct_words)} words from lyrics.txt.")
        else:
            print("lyrics.txt not found. Using transcribed lyrics.")
    elif use_genius:
        print("Fetching lyrics from Genius...")
        genius_url = "https://genius.com/Mili-jpn-worldexecuteme-lyrics"
        correct_lyrics = fetch_lyrics_from_genius(genius_url)
        if correct_lyrics:
            correct_words = [word.strip() for word in correct_lyrics.split() if word.strip()]
            print(f"Fetched {len(correct_words)} words from Genius.")
        else:
            print("Failed to fetch lyrics from Genius.")
    else:
        print("Using transcribed lyrics.")

    print("Transcribing audio...")
    transcription = transcribe_audio(str(audio_path))
    lyrics_segments = []
    for segment in transcription['segments']:
        if 'words' in segment:
            lyrics_segments.extend(segment['words'])
        else:
            lyrics_segments.append(segment)

    print(f"Transcribed {len(lyrics_segments)} segments.")

    # Replace lyrics with correct ones if available
    if correct_words:
        if len(correct_words) != len(lyrics_segments):
            print(f"Warning: Number of words in lyrics ({len(correct_words)}) does not match number of transcribed segments ({len(lyrics_segments)}). Timing may be off.")
        for i, segment in enumerate(lyrics_segments):
            if i < len(correct_words):
                segment['text'] = correct_words[i]
                segment['word'] = correct_words[i]

    print("Extracting pitch...")
    pitch_data = extract_pitch(str(audio_path))

    # Get audio duration
    y, sr = librosa.load(str(audio_path), sr=None)
    audio_duration = len(y) / sr

    ust_file = audio_path.parent / (audio_path.stem + ".ust")
    print(f"Generating UST file: {ust_file}")
    generate_ust(lyrics_segments, pitch_data, str(ust_file), audio_duration)

    print("Done! Open the UST file in OpenUTAU and render with DiffSinger.")

if __name__ == "__main__":
    main()

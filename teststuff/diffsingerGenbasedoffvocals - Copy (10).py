import whisper
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import gentle

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

def align_with_gentle(audio_path, text):
    """
    Use Gentle to force-align text to audio for better timing accuracy.
    """
    try:
        with open(audio_path, 'rb') as audio_file:
            resources = gentle.Resources()
            aligner = gentle.ForcedAligner(resources, text)
            result = aligner.transcribe(audio_file.read(), log=None)
            words = []
            for word in result.words:
                if word.case == 'success':
                    words.append({
                        'word': word.word,
                        'start': word.start,
                        'end': word.end
                    })
            return words
    except Exception as e:
        print(f"Gentle alignment failed: {e}. Falling back to Whisper timestamps.")
        return None

def extract_pitch(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr)
    hop_length = 256
    # Use advanced pitch estimation with better parameters
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), hop_length=hop_length, sr=sr, frame_length=2048, fill_na=np.nan)
    # Interpolate NaN values for smoother pitch
    f0 = librosa.util.fix_length(f0, size=len(y)//hop_length + 1)
    f0 = pd.Series(f0).interpolate().values
    return f0, voiced_flag, sr, hop_length

def detect_first_onset(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    if onset_frames.size > 0:
        return onset_frames[0]
    return 0.0

def detect_voiced_segments(audio_path, sr=22050, hop_length=256):
    """
    Detect voiced segments in the audio using librosa.
    Returns list of (start_time, end_time) tuples for voiced segments.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    # Use RMS energy to detect silence
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    # Threshold for silence (adjustable)
    silence_threshold = np.percentile(rms, 25)  # Lower 25% is considered silence

    # Find voiced frames
    voiced_frames = rms > silence_threshold

    # Convert to time segments
    voiced_segments = []
    start_frame = None
    for i, is_voiced in enumerate(voiced_frames):
        if is_voiced and start_frame is None:
            start_frame = i
        elif not is_voiced and start_frame is not None:
            end_frame = i
            start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
            end_time = librosa.frames_to_time(end_frame, sr=sr, hop_length=hop_length)
            voiced_segments.append((start_time, end_time))
            start_frame = None

    # Handle case where audio ends with voiced segment
    if start_frame is not None:
        end_time = librosa.frames_to_time(len(voiced_frames), sr=sr, hop_length=hop_length)
        voiced_segments.append((start_time, end_time))

    return voiced_segments



def adjust_ust_duration(ust_path, audio_duration):
    with open(ust_path, 'r', encoding='utf-8') as f:
        ust_content = f.read()

    lines = ust_content.split('\n')
    tempo = 120.0
    for line in lines:
        if line.startswith('Tempo='):
            tempo = float(line.split('=')[1])
            break

    ticks_per_second = 480 * tempo / 60

    total_ticks = 0
    note_lengths = []
    current_note = None
    for i, line in enumerate(lines):
        if line.startswith('[#') and line.endswith(']'):
            if current_note is not None:
                note_lengths.append(current_note)
            current_note = {'index': i, 'length': 0}
        elif line.startswith('Length='):
            if current_note:
                current_note['length'] = int(line.split('=')[1])
                total_ticks += current_note['length']

    if current_note:
        note_lengths.append(current_note)

    ust_duration = total_ticks / ticks_per_second
    if abs(ust_duration - audio_duration) > 0.1:
        scale_factor = audio_duration / ust_duration
        adjusted_content = ust_content
        for note in note_lengths:
            old_length = note['length']
            new_length = max(1, int(old_length * scale_factor))
            adjusted_content = adjusted_content.replace(f"Length={old_length}", f"Length={new_length}", 1)
        with open(ust_path, 'w', encoding='utf-8') as f:
            f.write(adjusted_content)
        print(f"Adjusted UST duration from {ust_duration:.2f}s to {audio_duration:.2f}s")

def trim_long_notes(ust_path, max_note_seconds=5.0, audio_duration=None):
    with open(ust_path, 'r', encoding='utf-8') as f:
        ust_content = f.read()

    lines = ust_content.split('\n')
    tempo = 120.0
    for line in lines:
        if line.startswith('Tempo='):
            tempo = float(line.split('=')[1])
            break

    ticks_per_second = 480 * tempo / 60
    max_ticks = int(max_note_seconds * ticks_per_second)
    audio_ticks = int(audio_duration * ticks_per_second) if audio_duration else None

    adjusted_content = ust_content
    cumulative_ticks = 0
    for i, line in enumerate(lines):
        if line.startswith('Length='):
            length = int(line.split('=')[1])
            if length > max_ticks:
                adjusted_content = adjusted_content.replace(f"Length={length}", f"Length={max_ticks}", 1)
                print(f"Trimmed note length from {length} to {max_ticks} ticks")
                length = max_ticks
            cumulative_ticks += length
            if audio_ticks and cumulative_ticks > audio_ticks:
                excess = cumulative_ticks - audio_ticks
                new_length = max(1, length - excess)
                adjusted_content = adjusted_content.replace(f"Length={length}", f"Length={new_length}", 1)
                print(f"Trimmed final note to fit audio duration: {length} to {new_length} ticks")
                break

    with open(ust_path, 'w', encoding='utf-8') as f:
        f.write(adjusted_content)

def generate_ust(lyrics_segments, pitch_data, output_path, audio_duration, from_gentle=False, first_onset=0.0, ustx_tempo=None):
    f0, voiced_flag, sr, hop_length = pitch_data
    tempo = ustx_tempo if ustx_tempo else 120  # Use USTX tempo if provided, else default
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

    # Use transcription times directly without scaling or shifting to copy exact word lengths
    if from_gentle:
        print("Using USTX-aligned times directly.")
    else:
        print("Using Whisper segment times directly.")

    note_index = 0
    previous_end = 0.0

    for segment in lyrics_segments:
        start_time = segment['start']
        end_time = segment['end']
        lyric = segment.get('text', segment.get('word', '')).strip()
        if not lyric:
            continue

        # Insert rest if there's a gap (only for non-USTX data)
        if not from_gentle:
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

        # Ensure minimum note length for audibility (e.g., 0.1 seconds)
        min_length_ticks = int(0.1 * ticks_per_second)
        if length < min_length_ticks:
            length = min_length_ticks
            print(f"Extended note '{lyric}' from {int(duration * ticks_per_second)} to {min_length_ticks} ticks for audibility.")

        # Use tone from USTX if available, otherwise calculate from pitch
        if 'tone' in segment:
            note_num = segment['tone']
        else:
            # Calculate average pitch
            start_frame = int(start_time * sr / hop_length)
            end_frame = int(end_time * sr / hop_length)
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

def validate_and_adjust_ust(ust_path, audio_duration):
    """
    Double-check the generated UST file for timing and note length issues.
    Adjust if necessary to match the audio duration.
    """
    with open(ust_path, 'r', encoding='utf-8') as f:
        ust_content = f.read()

    lines = ust_content.split('\n')
    tempo = 120.0  # Default, but should parse from UST
    for line in lines:
        if line.startswith('Tempo='):
            tempo = float(line.split('=')[1])
            break

    ticks_per_second = 480 * tempo / 60

    total_ticks = 0
    note_lengths = []
    note_indices = []
    current_note = None
    for i, line in enumerate(lines):
        if line.startswith('[#') and line.endswith(']'):
            if current_note is not None:
                note_lengths.append(current_note)
                note_indices.append(i - 1)  # Length line is before the note
            current_note = {'index': i, 'length': 0, 'lyric': ''}
        elif line.startswith('Length='):
            if current_note:
                current_note['length'] = int(line.split('=')[1])
                total_ticks += current_note['length']
        elif line.startswith('Lyric='):
            if current_note:
                current_note['lyric'] = line.split('=')[1]

    if current_note:
        note_lengths.append(current_note)

    ust_duration = total_ticks / ticks_per_second
    print(f"UST total duration: {ust_duration:.2f}s, Audio duration: {audio_duration:.2f}s")

    if abs(ust_duration - audio_duration) > 0.1:  # Allow 0.1s tolerance
        print("Timing mismatch detected. Adjusting note lengths to match audio duration.")
        scale_factor = audio_duration / ust_duration
        if scale_factor > 1.5 or scale_factor < 0.5:
            print(f"Warning: Large scale factor ({scale_factor:.2f}) indicates potential transcription issues.")
        adjusted_content = ust_content
        for note in note_lengths:
            old_length = note['length']
            new_length = max(1, int(old_length * scale_factor))  # Ensure minimum length
            # Find the Length line for this note
            for j in range(note['index'], len(lines)):
                if lines[j].startswith('Length='):
                    adjusted_content = adjusted_content.replace(f"Length={old_length}", f"Length={new_length}", 1)
                    break

        with open(ust_path, 'w', encoding='utf-8') as f:
            f.write(adjusted_content)
        print("UST file adjusted for timing.")
    else:
        print("Timing check passed.")

    # Check for excessively long notes
    max_note_length_seconds = 10.0  # Arbitrary threshold
    for note in note_lengths:
        length_seconds = note['length'] / ticks_per_second
        if length_seconds > max_note_length_seconds:
            print(f"Warning: Note '{note['lyric']}' at index {note['index']} is {length_seconds:.2f}s long, which may be too long.")

    # Additional check: Ensure no negative gaps or overlaps (though rests handle gaps)
    cumulative_ticks = 0
    for note in note_lengths:
        cumulative_ticks += note['length']
    # This is already checked by total_ticks, but we can add more granular checks if needed

def main():
    # Set to False to disable Genius lyric fetching and or you u justn wanna mess with this for fun <3
    use_genius = False
    # Set to True to use local lyrics from lyrics.txt instead of transcribed or Genius
    use_local_lyrics = False
    # Set to False to disable silence/sound detection filtering (useful for repeated lyrics)
    use_silence_detection = True

    # Look for vocals.wav in the script's directory
    script_dir = Path(__file__).parent
    audio_path = script_dir / "vocals.wav"

    print(f"Current directory: {Path.cwd()}")
    print(f"Looking for: {audio_path}")
    print(f"Exists: {audio_path.exists()}")

    if not audio_path.exists():
        print(f"Audio file {audio_path} not found.")
        return

    # Use advanced transcription and alignment for timing
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

    lyrics_segments = []
    if correct_words:
        print("Using Gentle for advanced forced alignment with provided lyrics...")
        text = ' '.join(correct_words)
        aligned_words = align_with_gentle(str(audio_path), text)
        if aligned_words:
            lyrics_segments = aligned_words
            print(f"Aligned {len(lyrics_segments)} words with Gentle.")
        else:
            print("Gentle alignment failed. Falling back to Whisper transcription.")
            transcription = transcribe_audio(str(audio_path))
            lyrics_segments = []
            for segment in transcription['segments']:
                if 'words' in segment:
                    lyrics_segments.extend(segment['words'])
                else:
                    lyrics_segments.append(segment)
            # Replace lyrics with correct ones
            if len(correct_words) != len(lyrics_segments):
                print(f"Warning: Number of words in lyrics ({len(correct_words)}) does not match number of transcribed segments ({len(lyrics_segments)}). Timing may be off.")
            for i, segment in enumerate(lyrics_segments):
                if i < len(correct_words):
                    segment['text'] = correct_words[i]
                    segment['word'] = correct_words[i]
    else:
        print("Transcribing audio with Whisper...")
        transcription = transcribe_audio(str(audio_path))
        lyrics_segments = []
        for segment in transcription['segments']:
            if 'words' in segment:
                lyrics_segments.extend(segment['words'])
            else:
                lyrics_segments.append(segment)

    print(f"Using {len(lyrics_segments)} segments.")

    # Detect voiced segments for silence/sound detection
    if use_silence_detection:
        print("Detecting voiced segments for silence/sound detection...")
        voiced_segments = detect_voiced_segments(str(audio_path))
        print(f"Detected {len(voiced_segments)} voiced segments.")

        # Filter lyrics segments to only include those within voiced segments
        filtered_lyrics_segments = []
        for segment in lyrics_segments:
            segment_start = segment['start']
            segment_end = segment['end']
            # Check if segment overlaps with any voiced segment
            for voice_start, voice_end in voiced_segments:
                if segment_start < voice_end and segment_end > voice_start:
                    # Overlap found, include this segment
                    filtered_lyrics_segments.append(segment)
                    break

        print(f"Filtered to {len(filtered_lyrics_segments)} segments within voiced regions.")
        lyrics_segments = filtered_lyrics_segments
    else:
        print("Silence/sound detection disabled. Using all transcribed segments.")

    print("Extracting pitch with advanced detection...")
    pitch_data = extract_pitch(str(audio_path))

    # Get audio duration
    y, sr = librosa.load(str(audio_path), sr=None)
    audio_duration = len(y) / sr

    # Detect first onset for timing alignment
    first_onset = detect_first_onset(str(audio_path))
    print(f"First onset detected at {first_onset:.2f}s")

    ust_file = audio_path.parent / (audio_path.stem + ".ust")
    print(f"Generating UST file: {ust_file}")
    generate_ust(lyrics_segments, pitch_data, str(ust_file), audio_duration, from_gentle=bool(aligned_words) if 'aligned_words' in locals() else False, first_onset=first_onset)

    print("Done! Open the UST file in OpenUTAU and render with DiffSinger.")

if __name__ == "__main__":
    main()

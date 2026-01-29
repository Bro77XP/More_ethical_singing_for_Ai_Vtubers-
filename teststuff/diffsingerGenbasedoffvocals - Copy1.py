import whisper
import librosa
import numpy as np
import json
from pathlib import Path

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result

def extract_pitch(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    return f0, voiced_flag, sr

def generate_ust(lyrics_segments, pitch_data, output_path):
    ust_content = """[#VERSION]
UST Version 1.2
[#SETTING]
Tempo=120.00
Tracks=1
ProjectName=Generated from Vocals
VoiceDir=%VOICE%\\Default
OutFile=
CacheDir=%CACHE%
Tool1=wavtool.exe
Tool2=resampler.exe
Mode2=True
[#0000]
Length=480
Lyric=R
NoteNum=60
PreUtterance=
VoiceOverlap=
Intensity=100
Modulation=0
Velocity=100
Flags=
"""
    note_num = 60  # Default note
    for i, segment in enumerate(lyrics_segments):
        start_time = segment['start']
        end_time = segment['end']
        lyric = segment['text'].strip()
        if not lyric:
            continue
        length = int((end_time - start_time) * 480 / 0.5)  # Rough estimate, 480 ticks per beat, assuming 120 BPM
        ust_content += f"[#{i+1:04d}]\n"
        ust_content += f"Length={length}\n"
        ust_content += f"Lyric={lyric}\n"
        ust_content += f"NoteNum={note_num}\n"
        ust_content += "PreUtterance=\n"
        ust_content += "VoiceOverlap=\n"
        ust_content += "Intensity=100\n"
        ust_content += "Modulation=0\n"
        ust_content += "Velocity=100\n"
        ust_content += "Flags=\n"
        note_num += 1  # Increment note for variety

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ust_content)

def main():
    audio_file = "(original-vocals) Test Voice 3 Erm, Ali V 8 Test Model 5 sings ᵇᵉˢᵗ ᶠʳⁱᴱᴺᴰ-2026-01-02.wav"
    audio_path = Path(audio_file)

    if not audio_path.exists():
        print(f"Audio file {audio_file} not found.")
        return

    print("Transcribing audio...")
    transcription = transcribe_audio(str(audio_path))
    lyrics_segments = transcription['segments']

    print("Extracting pitch...")
    f0, voiced_flag, sr = extract_pitch(str(audio_path))

    ust_file = audio_path.stem + ".ust"
    print(f"Generating UST file: {ust_file}")
    generate_ust(lyrics_segments, (f0, voiced_flag, sr), ust_file)

    print("Done! Open the UST file in OpenUTAU and render with DiffSinger.")

if __name__ == "__main__":
    main()

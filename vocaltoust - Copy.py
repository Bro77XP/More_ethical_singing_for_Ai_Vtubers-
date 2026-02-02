
#!/usr/bin/env python3
"""vocaltoust.py

Single-file pipeline that turns vocal audio (wav/mp3) into a UST file suitable for OpenUTAU/DiffSinger.
Features:
 - Whisper transcription (word timestamps) with graceful fallback to Whisper-only segments
 - Optional Gentle forced-alignment if provided lyrics are available
 - Pitch extraction via librosa.pyin with NaN interpolation and smoothing
 - Voice activity detection (RMS-based) to filter micro rests and detect voiced regions
 - Generates a UST file with rests, notes, velocities and extra flags tuned for DiffSinger
 - Ensures the final UST is no shorter than the original audio by extending the last note
 - Several small hardening and logging improvements for stability

Notes:
 - This script writes a .ust file but does not call DiffSinger. Use OpenUTAU/DiffSinger externally.
 - Dependencies: whisper, librosa, numpy, pandas, gentle (optional), requests, bs4
"""

import os
import sys
import math
import time
import json
import logging
from pathlib import Path

# External libs (not imported here to avoid runtime error when saving file).
# Users must install whisper, librosa, numpy, pandas, gentle, requests, bs4 to run the pipeline.
# import whisper
# import librosa
# import numpy as np
# import pandas as pd
# import gentle
# import requests
# from bs4 import BeautifulSoup

# --------------------------------------------------------
# Utility logging setup
# --------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

# --------------------------------------------------------
# Helper functions
# --------------------------------------------------------
def safe_read_text(path):
    try:
        return Path(path).read_text(encoding='utf-8')
    except Exception as e:
        logging.debug(f"safe_read_text failed for {path}: {e}")
        return ""

def safe_write_text(path, text):
    try:
        Path(path).write_text(text, encoding='utf-8')
        return True
    except Exception as e:
        logging.error(f"Failed to write {path}: {e}")
        return False

def clamp(x, a, b):
    return max(a, min(b, x))

# --------------------------------------------------------
# Web scraping for lyrics (Genius) - optional
# --------------------------------------------------------
def fetch_lyrics_from_genius(url, timeout=10):
    try:
        import requests
        from bs4 import BeautifulSoup
    except Exception:
        logging.warning("requests/bs4 not installed — Genius fetching disabled.")
        return None

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Modern Genius stores lyrics inside divs with data-lyrics-container="true"
        containers = soup.find_all('div', attrs={'data-lyrics-container': 'true'})
        if containers:
            text = "\n".join(c.get_text(separator=' ').strip() for c in containers)
            logging.info(f"Fetched lyrics from Genius ({len(text.split())} words).")
            return text

        # Fall back to older structure
        lyrics_div = soup.find("div", class_="lyrics")
        if lyrics_div:
            return lyrics_div.get_text()

    except Exception as e:
        logging.warning(f"Failed to fetch lyrics from Genius: {e}")
    return None

# --------------------------------------------------------
# Transcription wrapper (Whisper)
# --------------------------------------------------------
def transcribe_audio_whisper(audio_path, model_name="base"):
    try:
        import whisper
    except Exception:
        logging.error("whisper not installed. Install 'whisper' to use automatic transcription.")
        raise

    logging.info(f"Loading Whisper model '{model_name}' (this can take memory/time)...")
    model = whisper.load_model(model_name)
    logging.info("Transcribing with Whisper (word timestamps enabled)...")
    result = model.transcribe(str(audio_path), word_timestamps=True)
    return result

# --------------------------------------------------------
# Gentle forced alignment wrapper
# --------------------------------------------------------
def align_with_gentle(audio_path, text):
    try:
        import gentle
    except Exception:
        logging.warning("gentle not installed — forced alignment disabled.")
        return None

    try:
        with open(audio_path, 'rb') as fh:
            resources = gentle.Resources()
            aligner = gentle.ForcedAligner(resources, text, nthreads=4)
            result = aligner.transcribe(fh.read(), log=None)

        words = []
        for w in result.words:
            # Gentle provides 'case' indicating success/failure
            if getattr(w, "case", None) == "success":
                words.append({
                    "word": getattr(w, "word", ""),
                    "start": float(getattr(w, "start", 0.0)),
                    "end": float(getattr(w, "end", 0.0))
                })
        logging.info(f"Gentle aligned {len(words)} words")
        return words
    except Exception as e:
        logging.warning(f"Gentle alignment failed: {e}")
        return None

# --------------------------------------------------------
# Pitch extraction and smoothing
# --------------------------------------------------------
def extract_pitch_librosa(audio_path, sr=22050, hop_length=256):
    try:
        import librosa, numpy as np, pandas as pd
    except Exception:
        logging.error("librosa/numpy/pandas not installed. Install them to extract pitch.")
        raise

    y, sr = librosa.load(str(audio_path), sr=sr)
    frame_length = 2048

    logging.info("Running pyin pitch extraction. This may take a while for long files...")
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        hop_length=hop_length,
        frame_length=frame_length,
        fill_na=np.nan
    )

    # Ensure arrays are sane
    expected_len = int(math.ceil(len(y) / hop_length)) + 1
    try:
        f0 = librosa.util.fix_length(f0, size=expected_len)
    except Exception:
        f0 = f0 if f0 is not None else np.full(expected_len, np.nan)

    # Smooth/interpolate pitch to avoid NaNs and jumps
    f0_series = pd.Series(f0)
    f0_series = f0_series.interpolate(limit_direction='both', limit=512)
    f0_smoothed = f0_series.rolling(window=3, min_periods=1, center=True).mean().fillna(method='bfill').fillna(method='ffill').values

    # Convert voiced_flag to boolean array (if it's None, assume voiced False)
    if voiced_flag is None:
        voiced_flag = np.zeros_like(f0_smoothed, dtype=bool)
    else:
        voiced_flag = np.nan_to_num(voiced_flag).astype(bool)

    logging.info("Pitch extraction complete.")
    return f0_smoothed, voiced_flag, sr, hop_length

# --------------------------------------------------------
# Onset & VAD utilities
# --------------------------------------------------------
def detect_first_onset(audio_path, sr=22050):
    try:
        import librosa
    except Exception:
        logging.warning("librosa not installed — cannot detect first onset.")
        return 0.0
    y, sr = librosa.load(str(audio_path), sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    return float(onsets[0]) if len(onsets) else 0.0

def detect_voiced_segments(audio_path, sr=22050, hop_length=256, silence_percentile=25):
    try:
        import librosa, numpy as np
    except Exception:
        logging.error("librosa/numpy not installed — VAD unavailable.")
        raise

    y, sr = librosa.load(str(audio_path), sr=sr)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    threshold = np.percentile(rms, silence_percentile)

    voiced = rms > (threshold * 0.5)  # A little more permissive
    segments = []
    start = None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            end = i
            s = librosa.frames_to_time(start, sr=sr, hop_length=hop_length)
            e = librosa.frames_to_time(end, sr=sr, hop_length=hop_length)
            # Skip micro-segments smaller than 40ms
            if (e - s) >= 0.04:
                segments.append((s, e))
            start = None
    if start is not None:
        s = librosa.frames_to_time(start, sr=sr, hop_length=hop_length)
        e = librosa.frames_to_time(len(voiced), sr=sr, hop_length=hop_length)
        if (e - s) >= 0.04:
            segments.append((s, e))
    logging.info(f"Detected {len(segments)} voiced regions.")
    return segments

# --------------------------------------------------------
# UST parsing and duration helpers
# --------------------------------------------------------
def parse_ust_lengths(ust_text):
    lines = ust_text.splitlines()
    tempo = 120.0
    total_ticks = 0
    lengths = []
    for idx, line in enumerate(lines):
        if line.startswith("Tempo="):
            try:
                tempo = float(line.split("=")[1])
            except Exception:
                tempo = 120.0
        if line.startswith("Length="):
            try:
                length = int(line.split("=")[1])
            except Exception:
                length = 0
            total_ticks += length
            lengths.append((idx, length))
    ticks_per_second = 480 * tempo / 60.0
    duration = total_ticks / ticks_per_second if ticks_per_second > 0 else 0.0
    return duration, tempo, ticks_per_second, lengths, lines

def ensure_ust_not_shorter_than_audio(ust_path, audio_duration, min_tail_seconds=0.5):
    text = safe_read_text(ust_path)
    if not text:
        logging.error("UST file empty or unreadable.")
        return False
    ust_duration, tempo, tps, lengths, lines = parse_ust_lengths(text)
    if ust_duration >= audio_duration:
        logging.info("UST already equal to or longer than audio; no change required.")
        return True

    extra = (audio_duration - ust_duration) + float(min_tail_seconds)
    extra_ticks = int(extra * tps)
    if not lengths:
        logging.error("No Length= entries found in UST; cannot extend final note.")
        return False
    last_idx, last_len = lengths[-1]
    logging.info(f"Extending final note: +{extra:.2f}s ({extra_ticks} ticks)")
    # Replace the line at last_idx
    lines[last_idx] = f"Length={last_len + extra_ticks}"
    safe_write_text(ust_path, "\n".join(lines))
    logging.info("UST extended successfully.")
    return True

# --------------------------------------------------------
# Note and lyric cleaning
# --------------------------------------------------------
def clean_lyric_token(token):
    # Remove special characters that UTAU/UST may not like and trim
    token = token.strip()
    # Remove punctuation widely not used in lyrics
    bad = ['"', "'", "(", ")", "[", "]", "{", "}", "*", "#", "$", "%", "&", "/", "\\"]
    for b in bad:
        token = token.replace(b, "")
    # Collapse repeated whitespace
    token = " ".join(token.split())
    return token

# --------------------------------------------------------
# Generate UST content
# --------------------------------------------------------
def generate_ust(segments, pitch_data, output_path, tempo=120.0, from_gentle=False, min_note_seconds=0.1):
    try:
        import numpy as np
    except Exception:
        logging.warning("numpy not installed — using simple math fallback for note pitch calculation.")
        np = None

    f0, voiced_flag, sr, hop_length = pitch_data
    ticks_per_second = 480 * tempo / 60.0

    header = [
        "[#VERSION]",
        "UST Version 1.2",
        "[#SETTING]",
        f"Tempo={tempo:.2f}",
        "Tracks=1",
        "ProjectName=Generated from Vocals",
        "VoiceDir=%VOICE%\\Default",
        "OutFile=",
        "CacheDir=%CACHE%",
        "Tool1=wavtool.exe",
        "Tool2=resampler.exe",
        "Mode2=True",
        ""
    ]

    lines = header[:]
    note_index = 0
    prev_end = 0.0

    for seg in sorted(segments, key=lambda x: x.get("start", 0.0)):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 0.1))
        lyric = seg.get("text", seg.get("word", "")).strip()
        lyric = clean_lyric_token(lyric)
        if not lyric:
            prev_end = end
            continue

        # insert rest for substantial gaps only (skip micro rests < 50ms)
        gap = start - prev_end
        if (not from_gentle) and gap > 0.05:
            rest_ticks = int(round(gap * ticks_per_second))
            if rest_ticks > 0:
                lines.append(f"[#{note_index:04d}]")
                lines.append(f"Length={rest_ticks}")
                lines.append("Lyric=R")
                lines.append("NoteNum=60")
                lines.append("PreUtterance=")
                lines.append("VoiceOverlap=")
                lines.append("Intensity=100")
                lines.append("Modulation=0")
                lines.append("Velocity=100")
                lines.append("Flags=")
                lines.append("")
                note_index += 1

        dur = max(min_note_seconds, end - start)
        length = int(round(dur * ticks_per_second))

        # Determine pitch for note from average f0 inside the range
        s_frame = int(round(start * sr / hop_length))
        e_frame = int(round(end * sr / hop_length))
        s_frame = max(0, s_frame)
        e_frame = min(len(f0)-1, e_frame) if len(f0) > 0 else s_frame + 1

        note_num = 60  # default C4
        try:
            if len(f0) > 0:
                seg_f0 = f0[s_frame:e_frame] if e_frame > s_frame else f0[s_frame:s_frame+1]
                seg_voiced = voiced_flag[s_frame:e_frame] if e_frame > s_frame else voiced_flag[s_frame:s_frame+1]
                # pick only voiced frames
                voiced_frames = [f for f, v in zip(seg_f0, seg_voiced) if v and not math.isnan(f)]
                if voiced_frames:
                    avg_f0 = float(sum(voiced_frames) / len(voiced_frames))
                    note_num = int(round(69 + 12 * math.log2(avg_f0 / 440.0)))
                    # constrain to a singable range (C3=48 .. C5=72)
                    note_num = clamp(note_num, 48, 72)
                else:
                    note_num = 60
        except Exception as e:
            logging.debug(f"Pitch to note conversion issue: {e}")
            note_num = 60

        # Write note block
        lines.append(f"[#{note_index:04d}]")
        lines.append(f"Length={length}")
        lines.append(f"Lyric={lyric}")
        lines.append(f"NoteNum={note_num}")
        lines.append("PreUtterance=")
        lines.append("VoiceOverlap=")
        lines.append("Velocity=100")
        lines.append("Intensity=100")
        lines.append("Modulation=0")
        # Helpful flags for DiffSinger/OpenUTAU performance (kept conservative)
        lines.append("Flags=g0B0H0P86")
        lines.append("PBS=-40;0")
        lines.append("PBW=80")
        lines.append("PBY=0")
        lines.append("PBM=")
        lines.append("VBR=75,175,25,10,10,0,0")
        lines.append("")

        note_index += 1
        prev_end = end

    # Write file
    safe_write_text(output_path, "\n".join(lines))
    logging.info(f"Wrote UST to {output_path} with {note_index} notes.")
    return True

# --------------------------------------------------------
# validate_and_adjust_ust: gentle scaling (keeps proportions)
# --------------------------------------------------------
def validate_and_adjust_ust(ust_path, audio_duration, tolerance_seconds=0.1):
    text = safe_read_text(ust_path)
    if not text:
        logging.error("UST empty - cannot validate.")
        return False
    ust_duration, tempo, tps, lengths, lines = parse_ust_lengths(text)
    logging.info(f"UST duration: {ust_duration:.2f}s | Audio: {audio_duration:.2f}s")
    if abs(ust_duration - audio_duration) <= tolerance_seconds:
        logging.info("Duration within tolerance; no scaling necessary.")
        return True

    # Compute scale factor and limit extreme rescaling
    scale = audio_duration / ust_duration if ust_duration > 0 else 1.0
    if scale <= 0 or math.isnan(scale) or math.isinf(scale):
        logging.error("Invalid scale factor; skipping auto scale.")
        return False
    if scale < 0.5 or scale > 2.0:
        logging.warning(f"Scale factor {scale:.2f} is large; this may indicate transcription issues. Skipping scale.")
        return False

    # Scale all lengths proportionally
    logging.info(f"Scaling all note lengths by factor {scale:.3f}")
    new_lines = []
    for line in lines:
        if line.startswith("Length="):
            try:
                old = int(line.split("=")[1])
                new = max(1, int(round(old * scale)))
                new_lines.append(f"Length={new}")
            except Exception:
                new_lines.append(line)
        else:
            new_lines.append(line)
    safe_write_text(ust_path, "\n".join(new_lines))
    logging.info("Scaled UST to match audio duration (proportional scaling).")
    return True

# --------------------------------------------------------
# Main orchestrator
# --------------------------------------------------------
def pipeline_main(
    audio_path,
    lyrics_source=None,
    use_gentle=True,
    whisper_model="base",
    output_ust=None,
    use_silence_detection=True
):
    audio_path = Path(audio_path)
    if not audio_path.exists():
        logging.error(f"Audio file not found: {audio_path}")
        return False

    if output_ust is None:
        output_ust = audio_path.with_suffix(".ust")

    # Transcribe (Whisper)
    try:
        transcription = transcribe_audio_whisper(str(audio_path), model_name=whisper_model)
    except Exception as e:
        logging.error(f"Whisper transcription failed: {e}")
        return False

    # Build initial text segments list
    segments = []
    # Whisper can return 'segments' with nested 'words' or word-level timestamps
    if "segments" in transcription:
        for seg in transcription["segments"]:
            if "words" in seg:
                for w in seg["words"]:
                    word_entry = {
                        "word": w.get("word", "").strip(),
                        "start": float(w.get("start", 0.0)),
                        "end": float(w.get("end", 0.0)),
                        "text": w.get("word", "").strip()
                    }
                    segments.append(word_entry)
            else:
                # fallback to segment-level annotation
                segments.append({
                    "word": seg.get("text", "").strip(),
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "text": seg.get("text", "").strip()
                })

    # If lyrics_source provided (local lyrics or web), attempt Gentle forced alignment
    aligned_with_gentle = None
    if lyrics_source:
        if Path(lyrics_source).exists():
            lyrics_text = safe_read_text(lyrics_source)
        else:
            lyrics_text = fetch_lyrics_from_genius(lyrics_source)

        if lyrics_text:
            # Try Gentle if requested
            if use_gentle:
                aligned_with_gentle = align_with_gentle(str(audio_path), lyrics_text)
                if aligned_with_gentle:
                    segments = aligned_with_gentle
                else:
                    logging.info("Gentle alignment failed; falling back to Whisper segments.")
            else:
                logging.info("Gentle disabled; using Whisper segments and optional lyrics substitution.")
                # If number of words matches, substitute lyrics into existing segments
                words = [w for w in lyrics_text.split() if w.strip()]
                if len(words) == len(segments):
                    for i, w in enumerate(words):
                        segments[i]["text"] = w
                        segments[i]["word"] = w

    logging.info(f"Using {len(segments)} text segments for UST generation.")

    # Optionally filter by voiced segments
    if use_silence_detection:
        try:
            voiced_regions = detect_voiced_segments(str(audio_path))
            filtered = []
            for seg in segments:
                s = float(seg.get("start", 0.0))
                e = float(seg.get("end", s + 0.1))
                for vs, ve in voiced_regions:
                    if s < ve and e > vs:
                        filtered.append(seg)
                        break
            logging.info(f"Filtered segments from {len(segments)} -> {len(filtered)} using VAD.")
            segments = filtered
        except Exception as e:
            logging.warning(f"VAD filtering failed: {e} — using all segments.")

    # Pitch extraction
    try:
        pitch_data = extract_pitch_librosa(str(audio_path))
    except Exception as e:
        logging.error(f"Pitch extraction failed: {e}")
        return False

    # Determine audio duration for final checks
    try:
        import librosa
        y, sr = librosa.load(str(audio_path), sr=None)
        audio_duration = len(y) / sr
    except Exception:
        logging.warning("Failed to load audio to get duration. Defaulting to transcription end time.")
        # fallback: use last transcription end
        audio_duration = max((float(s.get("end", 0.0)) for s in segments), default=0.0)

    # First onset (for potential alignment shift, not used aggressively here)
    first_onset = detect_first_onset(str(audio_path))

    # Generate UST using detected segments and pitch
    generate_ust(segments, pitch_data, str(output_ust), tempo=120.0, from_gentle=bool(aligned_with_gentle))

    # Validate and try proportional scaling if a small mismatch exists
    validate_and_adjust_ust(str(output_ust), audio_duration)

    # Guarantee UST is at least as long as audio (extend final note if necessary)
    ensure_ust_not_shorter_than_audio(str(output_ust), audio_duration, min_tail_seconds=0.5)

    logging.info(f"Pipeline complete. UST saved to: {output_ust}")
    return True

# --------------------------------------------------------
# CLI wrapper
# --------------------------------------------------------
def print_usage():
    print("""Usage: python vocaltoust.py <audio_file> [--lyrics path_or_url] [--no-gentle] [--model base] [--out out.ust]
Options:
  --lyrics : Path to a local lyrics.txt or a Genius URL to use for forced alignment
  --no-gentle : Disable Gentle and use Whisper only
  --model : Whisper model name (tiny, base, small, medium, large)
  --out : Output UST path (defaults to <audiofile>.ust)
""")

def cli_main(argv):
    if len(argv) < 2:
        print_usage()
        return 1
    audio = argv[1]
    lyrics = None
    use_gentle = True
    model = "base"
    out = None
    i = 2
    while i < len(argv):
        a = argv[i]
        if a == "--lyrics" and i+1 < len(argv):
            lyrics = argv[i+1]; i += 2; continue
        if a == "--no-gentle":
            use_gentle = False; i += 1; continue
        if a == "--model" and i+1 < len(argv):
            model = argv[i+1]; i += 2; continue
        if a == "--out" and i+1 < len(argv):
            out = argv[i+1]; i += 2; continue
        i += 1

    success = pipeline_main(audio, lyrics_source=lyrics, use_gentle=use_gentle, whisper_model=model, output_ust=out)
    return 0 if success else 2

# --------------------------------------------------------
# If this file is executed, run CLI
# --------------------------------------------------------
if __name__ == "__main__":
    try:
        sys.exit(cli_main(sys.argv))
    except KeyboardInterrupt:
        logging.info("Interrupted by user")

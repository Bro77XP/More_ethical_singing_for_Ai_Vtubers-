# Vocal to UST Generator

This Python script (`vocaltoust.py`) generates a UST (UTAU Sequence Text) file from a vocal audio file (wav/mp3), suitable for use with DiffSinger in OpenUTAU. It transcribes the audio using Whisper, optionally aligns lyrics with Gentle, extracts pitch using Librosa, detects voiced regions, and creates a UST file with rests, notes, velocities, and flags tuned for DiffSinger.

## Features

- **Whisper Transcription**: Uses OpenAI Whisper for audio transcription with word-level timestamps.
- **Optional Gentle Alignment**: If lyrics are provided (local file or Genius URL), uses Gentle for forced alignment; falls back to Whisper segments.
- **Pitch Extraction**: Uses Librosa pyin for pitch detection with NaN interpolation and smoothing.
- **Voice Activity Detection**: Filters micro rests and detects voiced regions using RMS-based VAD.
- **UST Generation**: Creates UST files with rests, notes, velocities, and DiffSinger-optimized flags.
- **Duration Matching**: Ensures the UST is at least as long as the audio by extending the last note.
- **Flexible Input**: Supports local lyrics files or Genius URLs for lyrics.

## Requirements

- Python 3.7+
- Libraries:
  - openai-whisper
  - librosa
  - numpy
  - pandas
  - gentle (optional, for forced alignment)
  - requests
  - beautifulsoup4

Install the required libraries using pip:

```bash
pip install openai-whisper librosa numpy pandas gentle requests beautifulsoup4
```

## Usage

Run the script from the command line with the following options:

```bash
python vocaltoust.py --audio <audio_file> [--lyrics <lyrics_source>] [--no-gentle] [--model <whisper_model>] [--out <output_ust>]
```

### Options

- `--audio <audio_file>`: Path to the vocal audio file (wav/mp3). Required.
- `--lyrics <lyrics_source>`: Path to a local lyrics.txt file or a Genius URL for forced alignment. Optional.
- `--no-gentle`: Disable Gentle forced alignment and use Whisper only. Optional.
- `--model <whisper_model>`: Whisper model name (tiny, base, small, medium, large). Default: base. Optional.
- `--out <output_ust>`: Output UST file path. Defaults to `<audio_file>.ust`. Optional.

### Examples

1. **Basic usage with audio file**:
   ```bash
   python vocaltoust.py --audio vocals.wav
   ```

2. **With local lyrics**:
   ```bash
   python vocaltoust.py --audio vocals.wav --lyrics lyrics.txt
   ```

3. **With Genius URL**:
   ```bash
   python vocaltoust.py --audio vocals.wav --lyrics "https://genius.com/Artist-Song-lyrics"
   ```

4. **Custom output and model**:
   ```bash
   python vocaltoust.py --audio vocals.wav --lyrics lyrics.txt --model small --out output.ust
   ```

5. **Disable Gentle**:
   ```bash
   python vocaltoust.py --audio vocals.wav --lyrics lyrics.txt --no-gentle
   ```

### Output

- The script generates a `.ust` file (default: `<audio_file>.ust`) compatible with OpenUTAU and DiffSinger.
- Open the UST in OpenUTAU to render with DiffSinger.

## How It Works

1. **Transcription**: Transcribes audio using Whisper with word timestamps.
2. **Alignment**: If lyrics provided, attempts Gentle forced alignment; otherwise uses Whisper segments.
3. **Filtering**: Applies VAD to filter segments to voiced regions.
4. **Pitch Extraction**: Extracts pitch using Librosa pyin with smoothing.
5. **UST Creation**: Generates UST with notes, rests, pitch, and flags.
6. **Validation**: Scales and extends UST to match audio duration.

## Troubleshooting

- **Audio File Not Found**: Ensure the path to `--audio` is correct and the file exists.
- **Library Errors**: Install all required libraries.
- **Gentle Alignment Fails**: Falls back to Whisper; check Gentle installation.
- **UST Issues**: Verify in OpenUTAU; ensure audio is vocal-only.
- **Path Issues**: Use absolute paths if relative paths fail.

## License

This script is provided as-is for educational and personal use.


to run expermental do:
 python vocaltoust.py --audio "C:\Users\...\Downloads\More_ethical_singing_for_Ai_Vtubers--Lyricgen\More_ethical_singing_for_Ai_Vtubers--Lyricgen\teststuff\vocals.wav" --lyrics "C:\Users\...\Downloads\More_ethical_singing_for_Ai_Vtubers--Lyricgen\More_ethical_singing_for_Ai_Vtubers--Lyricgen\teststuff\lyrics.txt" --output "output.ust"

 you may have to change ... to the user path of what
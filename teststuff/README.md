# DiffSinger UST Generator from Vocals

This Python script generates a UST (UTAU Sequence Text) file from a vocal audio file, suitable for use with DiffSinger in OpenUTAU. It transcribes the audio using Whisper, extracts pitch using advanced Librosa techniques, and creates a UST file with lyrics, timing, and pitch data.

## Features

- **Advanced Audio Transcription**: Uses OpenAI Whisper with word-level timestamps for precise timing.
- **Advanced Pitch Extraction**: Uses Librosa with enhanced parameters including interpolation for smoother pitch detection.
- **UST Generation**: Creates a UST file compatible with OpenUTAU and DiffSinger.
- **Pitch Constrained**: Pitch is constrained to the range F#3 (54) to A#4 (70) for vocal suitability.
- **Timing Corrections**: Improved timing to avoid long initial silences and ensure accurate note lengths.
- **Long Note Support**: Supports long notes in OpenUTAU via adjustable Length fields.
- **Optional Lyric Fetching**: Can fetch accurate lyrics from Genius.com to replace transcribed lyrics.
- **Disablable Genius Fetching**: Lyric fetching from Genius can be disabled by setting `use_genius = False`.
- **Local Lyrics Support**: Can use lyrics from a local `lyrics.txt` file by setting `use_local_lyrics = True`.
- **Advanced Detection**: Uses Gentle for forced alignment when lyrics are provided, falling back to Whisper transcription.

## Requirements

- Python 3.7+
- Libraries:
  - whisper
  - librosa
  - numpy
  - pandas
  - requests
  - beautifulsoup4
  - gentle

Install the required libraries using pip:

```bash
pip install openai-whisper librosa numpy pandas requests beautifulsoup4 gentle
```

## Usage

1. **Prepare the Audio File**:
   - Rename your vocal audio file to `vocals.wav`.
   - Place `vocals.wav` in the same directory as the script (`diffsingerGenbasedoffvocals.py`and make sure to keep both in the same folder).

2. **Run the Script**:
   - Open a terminal or command prompt.
   - Navigate to the directory containing the script.
   - Run the script:

     ```bash
     python diffsingerGenbasedoffvocals.py
     ```

3. **Output**:
   - The script will generate `vocals.ust` in the same directory.
   - Open `vocals.ust` in OpenUTAU and render with DiffSinger.

## Configuration

- **Disable Genius Lyric Fetching**:
  - In the script, change `use_genius = True` to `use_genius = False` in the `main()` function.
  - This will skip fetching lyrics from Genius.com and use only the transcribed lyrics.

- **Use Local Lyrics**:
  - Set `use_local_lyrics = True` in the `main()` function to use lyrics from a `lyrics.txt` file in the script's directory.
  - If `lyrics.txt` is not found, it falls back to transcribed lyrics.
  - This takes priority over Genius fetching if both are enabled.

- **Genius URL**:
  - The script is hardcoded to fetch lyrics from a specific Genius URL.
  - To use a different song, modify the `genius_url` variable in the `main()` function.

## How It Works

1. **Transcription**: The script uses Whisper to transcribe the audio and extract word-level timestamps.
2. **Alignment**: If lyrics are provided, Gentle is used for forced alignment to improve timing accuracy.
3. **Pitch Extraction**: Librosa extracts the pitch (fundamental frequency) from the audio with advanced parameters and interpolation.
4. **Lyric Correction (Optional)**: If enabled, fetches lyrics from Genius.com and replaces transcribed words with accurate lyrics.
5. **UST Generation**: Combines transcription, pitch, and timing data into a UST file format.
   - Notes are created for each word with appropriate length, lyric, and pitch.
   - Pitch is constrained to a vocal range.
   - Timing is adjusted to start from the first lyric and avoid excessive silences.
   - This allows for quick time-saving while maintaining high accuracy for AI VTuber karaoke and ethical singing systems (avoiding RVC).

## Troubleshooting

- **Audio File Not Found**: Ensure the file is named `vocals.wav` and is in the script's directory.
- **Library Errors**: Ensure all required libraries are installed.
- **Genius Fetching Fails**: Check internet connection or disable Genius fetching.
- **UST File Issues**: Open the generated UST in OpenUTAU to verify.
- Please use the voice bank provided for RVC conversion as you may deal with legal issues with other virtual singers.

## License

This script is provided as-is for educational and personal use.

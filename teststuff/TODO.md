# TODO for Fixing Timing and Note Length Issues in DiffSinger UST Generator

## Completed Tasks
- [x] Fixed pitch extraction bug: Removed incorrect addition of `first_start` in frame calculations.
- [x] Enhanced `validate_and_adjust_ust` function:
  - Added lyric tracking for better warnings.
  - Added warning for large scale factors indicating transcription issues.
  - Ensured minimum note length of 1 tick.
  - Improved warning messages for long notes.
- [x] Removed duplicate `main()` call at the end of the script.

## Pending Tasks
- [ ] Test the script with a sample vocals.wav file to verify timing improvements.
- [ ] If issues persist, consider using a more accurate transcription model or manual alignment for long songs.

## Notes
- The uniform scaling in `generate_ust` may not fully correct cumulative drift from Whisper transcription errors in longer audio.
- For better accuracy, consider segment-wise alignment or using forced alignment tools like Gentle.

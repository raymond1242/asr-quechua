import os
import parselmouth
from parselmouth.praat import call

# --- Configuration ---
INPUT_FOLDER = "wav_audios"
OUTPUT_FOLDER = "segmented_audios"

# To TextGrid (silences) parameters
PITCH_FLOOR = 100        # Hz
TIME_STEP = 0.0          # 0.0 = auto
SILENCE_THRESHOLD = -49  # dB
MIN_SILENCE_DURATION = 0.25  # seconds
MIN_SOUNDING_DURATION = 0.2  # seconds


def _format_ts(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for filename in sorted(os.listdir(INPUT_FOLDER)):
        if not filename.lower().endswith(".wav"):
            continue

        filepath = os.path.join(INPUT_FOLDER, filename)
        sound = parselmouth.Sound(filepath)
        name = os.path.splitext(filename)[0]

        # Create a subfolder per audio file
        audio_out_folder = os.path.join(OUTPUT_FOLDER, name)
        os.makedirs(audio_out_folder, exist_ok=True)

        # Detect silent/sounding intervals via Praat's "To TextGrid (silences)"
        textgrid = call(
            sound, "To TextGrid (silences)",
            PITCH_FLOOR, TIME_STEP, SILENCE_THRESHOLD,
            MIN_SILENCE_DURATION, MIN_SOUNDING_DURATION,
            "silent", "sounding",
        )

        # Extract sounding segments and build SRT entries
        n_intervals = call(textgrid, "Get number of intervals...", 1)
        segment_idx = 1
        srt_lines = []

        for i in range(1, n_intervals + 1):
            label = call(textgrid, "Get label of interval...", 1, i)
            if label != "sounding":
                continue

            start = call(textgrid, "Get start time of interval...", 1, i)
            end = call(textgrid, "Get end time of interval...", 1, i)
            segment = sound.extract_part(start, end)

            out_path = os.path.join(audio_out_folder, f"{name}_seg{segment_idx:03d}.wav")
            segment.save(out_path, "WAV")

            srt_lines.append(f"{segment_idx}\n{_format_ts(start)} --> {_format_ts(end)}\n\n")
            segment_idx += 1

        # Write SRT file
        srt_path = os.path.join(audio_out_folder, f"{name}.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.writelines(srt_lines)

        print(f"{filename}: {segment_idx - 1} segments")

    print("Processing completed.")


if __name__ == "__main__":
    main()

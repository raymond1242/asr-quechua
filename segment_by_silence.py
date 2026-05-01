import os
import parselmouth
from parselmouth.praat import call

# --- Configuration ---
INPUT_FOLDER = "wav_audios"
OUTPUT_FOLDER = "segmented_audios_auto"

# To TextGrid (silences) parameters
PITCH_FLOOR = 100        # Hz
TIME_STEP = 0.0          # 0.0 = auto
MIN_SILENCE_DURATION = 0.25  # seconds
MIN_SOUNDING_DURATION = 0.55  # seconds

# Adaptive silence threshold: dB above the noise floor (10th percentile of intensity).
# Final threshold is computed per audio, relative to its own peak intensity.
NOISE_FLOOR_MARGIN_DB = 3.0

# Maximum segment duration. Longer sounding intervals are re-split.
MAX_SEGMENT_DURATION = 30.0  # seconds
# Aggressive parameters used when re-splitting long segments.
RESPLIT_MIN_SILENCE_DURATION = 0.1
RESPLIT_MIN_SOUNDING_DURATION = 0.2
RESPLIT_THRESHOLD_BUMP_DB = 5.0  # raise threshold (less strict) to catch shorter pauses


def _adaptive_silence_threshold(sound):
    """Compute a silence threshold (dB, relative to peak) from the audio's own
    intensity distribution: a few dB above the 10th-percentile noise floor."""
    intensity = sound.to_intensity(minimum_pitch=PITCH_FLOOR)
    peak_db = call(intensity, "Get maximum...", 0, 0, "Parabolic")
    noise_floor_db = call(intensity, "Get quantile...", 0, 0, 0.1)
    print(f"Peak: {peak_db}, Noise floor: {noise_floor_db}")
    return (noise_floor_db + NOISE_FLOOR_MARGIN_DB) - peak_db


def _split_long_interval(sound, start, end, base_threshold):
    """Split a sounding interval longer than MAX_SEGMENT_DURATION into shorter
    pieces. Tries aggressive silence detection first, then falls back to
    even-sized force splits when no internal silences are found."""
    duration = end - start
    if duration <= MAX_SEGMENT_DURATION:
        return [(start, end)]

    sub = sound.extract_part(start, end, preserve_times=False)
    threshold = base_threshold + RESPLIT_THRESHOLD_BUMP_DB

    try:
        sub_tg = call(
            sub, "To TextGrid (silences)",
            PITCH_FLOOR, TIME_STEP, threshold,
            RESPLIT_MIN_SILENCE_DURATION, RESPLIT_MIN_SOUNDING_DURATION,
            "silent", "sounding",
        )
        n = call(sub_tg, "Get number of intervals...", 1)
        sub_intervals = []
        for i in range(1, n + 1):
            label = call(sub_tg, "Get label of interval...", 1, i)
            if label != "sounding":
                continue
            s = call(sub_tg, "Get start time of interval...", 1, i) + start
            e = call(sub_tg, "Get end time of interval...", 1, i) + start
            sub_intervals.append((s, e))
    except Exception:
        sub_intervals = []

    # If aggressive splitting didn't help, force even-sized splits.
    if not sub_intervals or all((e - s) > MAX_SEGMENT_DURATION for s, e in sub_intervals):
        n_pieces = int(duration // MAX_SEGMENT_DURATION) + 1
        piece = duration / n_pieces
        return [(start + i * piece, start + (i + 1) * piece) for i in range(n_pieces)]

    # Recurse on any sub-intervals still too long.
    result = []
    for s, e in sub_intervals:
        if (e - s) > MAX_SEGMENT_DURATION:
            result.extend(_split_long_interval(sound, s, e, threshold))
        else:
            result.append((s, e))
    return result


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

        # Adaptive silence threshold per audio (dB relative to peak intensity)
        silence_threshold = _adaptive_silence_threshold(sound)

        # Detect silent/sounding intervals via Praat's "To TextGrid (silences)"
        textgrid = call(
            sound, "To TextGrid (silences)",
            PITCH_FLOOR, TIME_STEP, silence_threshold,
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

            for s, e in _split_long_interval(sound, start, end, silence_threshold):
                segment = sound.extract_part(s, e)
                out_path = os.path.join(audio_out_folder, f"{name}_seg{segment_idx:03d}.wav")
                segment.save(out_path, "WAV")

                srt_lines.append(f"{segment_idx}\n{_format_ts(s)} --> {_format_ts(e)}\n\n")
                segment_idx += 1

        # Write SRT file
        srt_path = os.path.join(audio_out_folder, f"{name}.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.writelines(srt_lines)

        print(f"{filename}: {segment_idx - 1} segments.\nSilence threshold: {silence_threshold}")

    print("Processing completed.")


if __name__ == "__main__":
    main()

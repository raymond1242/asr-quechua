import argparse
from pathlib import Path

from metrics import compute_cer
from transcribe import transcribe


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Omnilingual ASR")
    parser.add_argument("audio", help="Path to the WAV audio file (e.g. Quechua_Collao_Corpus/Audios/10006.wav)")
    parser.add_argument("--lang", default="que_Latn", help="Language code (default: que_Latn)")
    parser.add_argument(
        "--model",
        # default="omniASR_LLM_1B_v2",   # Not working for out of memory error, need to use smaller model
        # default="omniASR_CTC_300M_v2", # This model is smaller and works
        default="omniASR_CTC_1B_v2",     # This model is smaller and works
        help="Model card name (default: omniASR_LLM_1B_v2)",
    )
    parser.add_argument(
        "--reference",
        metavar="TEXT_OR_FILE",
        default=None,
        help="Texto de referencia (ground truth) o ruta a archivo .txt para evaluar CER",
    )
    args = parser.parse_args()

    transcription = transcribe(args.audio, lang=args.lang, model_card=args.model)
    print(transcription)

    if args.reference is not None:
        ref = args.reference.strip()
        ref_path = Path(ref)
        if ref_path.exists() and ref_path.is_file():
            ref = ref_path.read_text(encoding="utf-8").strip()
        cer = compute_cer(reference=ref, hypothesis=transcription)
        print(f"CER: {cer:.4f}")


if __name__ == "__main__":
    main()

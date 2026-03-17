import argparse

from transcribe import transcribe


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using Omnilingual ASR"
    )
    parser.add_argument(
        "audio",
        help="Path to the audio file (.wav, .flac, etc.)",
    )
    parser.add_argument(
        "--model",
        default="omniASR_CTC_300M",
        help="Model card name (default: omniASR_LLM_Unlimited_7B_v2)",
    )
    parser.add_argument(
        "--lang",
        default="que_Latn",
        help="Language code (default: que_Latn)",
    )
    args = parser.parse_args()

    transcription = transcribe(
        args.audio,
        lang=args.lang,
        model_card=args.model,
    )
    print(transcription)


if __name__ == "__main__":
    main()

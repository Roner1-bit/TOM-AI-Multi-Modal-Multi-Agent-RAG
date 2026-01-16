import whisper


def main():
    model = whisper.load_model("base")
    result = model.transcribe("mic_test.wav")
    print(result["text"].strip())


if __name__ == "__main__":
    main()

import sounddevice as sd
import soundfile as sf


def main():
    dev = sd.query_devices(sd.default.device[0], "input")
    sr = int(dev.get("default_samplerate", 44100))
    seconds = 3

    print("Default input device:", dev["name"])
    print("Sample rate:", sr)

    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()

    sf.write("mic_test.wav", audio, sr)
    print("Saved mic_test.wav")


if __name__ == "__main__":
    main()

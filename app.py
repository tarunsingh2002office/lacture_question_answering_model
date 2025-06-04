import whisper
import time

if __name__ == "__main__":
    start_time = time.time()
    model = whisper.load_model("turbo")
    load_time = time.time() - start_time  # Time taken to load the model
    print(f"Model loaded in {load_time:.2f} seconds.")
    start_transcription = time.time()
    result = model.transcribe("1_audio.mp3")
    transcription_time = time.time() - start_transcription  # Time taken for transcription
    print(f"Transcription completed in {transcription_time:.2f} seconds.")
    print(result["text"])

    


    # audio = whisper.load_audio("a.mp3")
    # print("2")
    # audio = whisper.pad_or_trim(audio)
    # print("3")
    # mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    # print("4")
    # print(f"#################========{mel}")
    # # detect the spoken language
    # _, probs = model.detect_language(mel)
    # print(f"Detected language: {max(probs, key=probs.get)}")
    # # result = model.transcribe("a.mp3")
    # # print(result["text"])
    # # decode the audio
    # options = whisper.DecodingOptions()
    # result = whisper.decode(model, mel, options)
    # print("--------------------------------------")
    # # print the recognized text
    # print(result.text)
    # print("--------------------------------------")
    # print("hello")
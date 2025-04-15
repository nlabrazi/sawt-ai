import whisper
import contextlib
import io

# def transcribe_audio(AUDIO_PATH: str, model_size: str = "medium") -> list:
#     model = whisper.load_model(model_size)
#     result = model.transcribe(AUDIO_PATH, language="ar", verbose=False)
#     return result["segments"]

def transcribe_audio(AUDIO_PATH: str, model_size="medium"):
    model = whisper.load_model(model_size)
    with contextlib.redirect_stdout(io.StringIO()):
        result = model.transcribe(AUDIO_PATH, language="ar", verbose=False)
    return result["segments"]

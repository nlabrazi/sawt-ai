import whisper
import contextlib
import io

def transcribe_audio(AUDIO_PATH: str, model_size="medium"):
    """
    Transcrit le fichier audio complet avec Whisper.
    Utilise word_timestamps pour affichage progressif.
    """
    model = whisper.load_model(model_size)

    with contextlib.redirect_stdout(io.StringIO()):
        result = model.transcribe(
            AUDIO_PATH,
            language="ar",
            verbose=False,
            word_timestamps=True
        )

    return result["segments"]

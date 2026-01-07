from transformers import pipeline

def load_model():
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
    return pipe

def generate(model, sample):
    src = sample["src_lang"]
    tgt = sample["tgt_lang"]
    task = "transcribe" if src == tgt else "translate"
    try:
        transcriptions = model(sample["sample"], generate_kwargs={"language": src, "task": task}, 
                        chunk_length_s=30, stride_length_s=(5,5), return_timestamps="word")
    except (TypeError, IndexError):
        # mandi/audio/zh/170.wav causes an error if return_timestamps="word" but not with None.
        # We don't change it for all because that would give different outputs and make us to 
        # re-generate many files :(
        transcriptions = model(sample["sample"], generate_kwargs={"language": src, "task": task}, 
                        chunk_length_s=30, stride_length_s=(5,5), 
                        return_timestamps=None)
    return transcriptions['text']

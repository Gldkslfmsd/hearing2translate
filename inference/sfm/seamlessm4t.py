from transformers import AutoProcessor, SeamlessM4Tv2Model
import librosa

def load_model():
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    return model, processor

seamless_lang_codes = {
    "en": "eng",
    "es": "spa",       
    "fr": "fra",
    "de": "deu",
    "it": "ita",
    "pt": "por",
    "zh": "cmn",
    "zh-cn": "cmn",
    "nl": "nld",
}

def generate(model, sample):
    model, processor = model
    src = seamless_lang_codes[sample["src_lang"]]
    tgt = seamless_lang_codes[sample["tgt_lang"]]

    audio, sr = librosa.load(sample["sample"], sr=processor.feature_extractor.sampling_rate)
    inputs = processor(audios=audio, return_tensors="pt")

    out = model.generate(**inputs, tgt_lang=tgt, generate_speech=False)[0].cpu().numpy().squeeze()
    text = processor.decode(out, skip_special_tokens=True)
    return text

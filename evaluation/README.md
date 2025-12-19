## Installation

### Set environment variables

Before running any scripts, you must configure the base path for the project and your Hugging Face cache settings.

```bash
# MANDATORY: Set the base path to the hearing2translate repository
export BASE_PATH="/path_to/hearing2translate" 

# Hugging Face Configuration
export HUGGINGFACE_TOKEN=''          # your personal HF token
export HF_HOME="/path/to/hf_cache"   # base directory for HF cache
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export DATASETS_CACHE="$HF_HOME/datasets"
```

---

### Handling Unbabel Models

When using `Unbabel` models (e.g., `Unbabel/wmt22-cometkiwi-da`, `Unbabel/XCOMET-XL`, `Unbabel/XCOMET-XXL`), the code will automatically attempt to download them from Hugging Face.

⚠️ **Important**: Many of these models require requesting access on Hugging Face. If you try to load them without an access token, the download will fail.

To avoid this:

1. Request access to the model from its Hugging Face page.
2. Once access is granted, log in to Hugging Face with your token:

```bash
huggingface-cli login
```

or, if you prefer environment variables:

```bash
huggingface-cli login --token $HUGGINGFACE_TOKEN
```

### Download XComet-XXL model

```python
from comet import download_model, load_from_checkpoint
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_path = download_model("Unbabel/XCOMET-XXL")
print("Unbabel/XCOMET-XXL model path:", model_path)

# download XLM-Roberta-XXL model in HF CACHE
tokenizer = AutoTokenizer.from_pretrained('facebook/xlm-roberta-xxl')
model = AutoModelForMaskedLM.from_pretrained("facebook/xlm-roberta-xxl")
```

---

### Download MetricX24-XXL Model

The [MetricX-24 models](https://huggingface.co/google/metricx-24-hybrid-large-v2p6) are hosted by Google on Hugging Face. They require both a checkpoint and a tokenizer (usually `mt5-xl`-based).

```python
from huggingface_hub import snapshot_download

model_name = "google/metricx-24-hybrid-xxl-v2p6-bfloat16"
model_path = snapshot_download(repo_id=model_name)
print("MetricX-24-XXL model is downloaded to:", model_path)
```

```python
from huggingface_hub import snapshot_download

model_name = "google/mt5-xxl"
model_path = snapshot_download(repo_id=model_name, ignore_patterns=['tf_model.h5', 'pytorch_model.bin'])

print("T5 tokenizer model is downloaded to:", model_path)
```

### Set Model Paths as env variables

After downloading the necessary models, you need to set their paths as environment variables. All models are stored in $HF_HOME cache folder:

```bash
export METRICX_CK_NAME='${HF_HOME}/hub/models--google--metricx-24-hybrid-xxl-v2p6/snapshots/0ff238ccb517eb0b2998dd6d299528b040c5caec' 
export METRICX_TOKENIZER='${HF_HOME}/hub/models--google--mt5-xxl/snapshots/e07c395916dfbc315d4e5e48b4a54a1e8821b5c0'
export XCOMET_CK_NAME='${HF_HOME}/hub/models--Unbabel--XCOMET-XXL/snapshots/873bac1b1c461e410c4a6e379f6790d3d1c7c214/checkpoints/model.ckpt'
```

### Output File Structure

All evaluation results are saved under the base directory defined in the scripts as `SAVING_BASE_DIR` (set to ./output_evals/). The structure is standardized across all benchmarks:

```
./output_evals
└── <BENCHMARK_NAME> 
    └── <SYSTEM_NAME> 
        └── <LANGUAGE_PAIR>
            ├── results.jsonl         # Detailed, per-item evaluation results
            └── results_summary.jsonl # Aggregated, final summary metrics
```

Where:

- <BENCHMARK_NAME> is the name of the dataset (e.g., fleurs, acl6060-long).
- <SYSTEM_NAME> is the model being evaluated (e.g., phi4multimodal, tower_whisper).
- <LANGUAGE_PAIR> is the language direction.

### Benchmarks

To run the evaluations, execute the corresponding bash script for the desired benchmark. Ensure `BASE_PATH` is set before running any of the following scripts.

#### Fleurs

```bash
bash run_evals_fleurs.sh
```

#### CoVoST2

```bash
bash run_evals_covost2.sh
```

#### EuroParl-ST

```bash
bash run_evals_europarl.sh
```

#### WMT

```bash
bash run_evals_wmt.sh
```

#### WinoST

```bash
# install spacy library
pip install spacy
# install stanza library
pip install stanza

# Spacy models need to have been previously been downloaded
for x in de_core_news_sm fr_core_news_sm es_core_news_sm it_core_news_sm pt_core_news_sm; do python3 -m spacy download $x; done

bash run_evals_winoST.sh
```

#### CommonAccent

```bash
bash run_evals_commonAccent.sh
```

#### ManDi

```bash
bash run_evals_mandi.sh
```

#### CS-Dialogue

```bash
bash run_evals_cs_dialog.sh
```

#### CS-Fleurs

```bash
bash run_evals_cs_fleurs.sh
```

#### LibriStutter

```bash
bash run_evals_libristutter.sh
```

#### NEuRoparl-ST

Metric calculations script is extracted from the Neuroparl-ST release.
Example of usage:

```bash
#Spacy models need to have been previously been downloaded.
for x in en_core_web_lg es_core_news_lg fr_core_news_lg it_core_news_lg; do python3 -m spacy download $x; done

SRC_CODE=en
TGT_CODE=es
SRCTGT="$SRC_CODE"_"$TGT_CODE"
MODEL=canary-v2
python3 ${BASE_PATH}/evaluation/metrics/neuroparl_st/ne_terms_accuracy.py --input <(jq -r '.output' ${BASE_PATH}/evaluation/output_evals/europarl_st/$MODEL/$SRCTGT/results.jsonl) --tsv-ref ${BASE_PATH}/evaluation/metrics/neuroparl_st/terms_ne_files/$SRCTGT/test.all.$TGT_CODE.iob --lang $TGT_CODE --save_dir ${BASE_PATH}/evaluation/output_evals/neuroparl_st/$MODEL/$SRCTGT/
```

#### NoisyFleurs

```bash
# ambient noise
bash run_evals_noisy_fleurs_ambient.sh
# babble noise
bash run_evals_noisy_fleurs_babble.sh
```

#### mExpresso

```bash
bash run_evals_mexpresso.sh
```

#### EmotionTalk

```bash
bash run_evals_emotiontalk.sh
```

#### ACL6060

This benchmark is split into Short and Long contexts. For ACL6060 Long, the evaluation requires segmenting the long benchmarks first. This process depends on the NLLB tokenizer. The NLLB tokenizer is required for segmenting long-context benchmarks using `mweralign`. You can download the sentencepiece model directly using wget:

```
wget https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/sentencepiece.bpe.model
```


```bash
export NLLB_TOKENIZER=''

# Run Short context evaluations
bash run_evals_acl6060_short.sh

# Run Long context evaluations (Requires NLLB_TOKENIZER)
bash run_evals_acl6060_long.sh
```

#### MCIF

This benchmark is split into Short and Long contexts. For MCIF Long, the evaluation requires segmenting the long benchmarks first. This process depends on the NLLB tokenizer. The NLLB tokenizer is required for segmenting long-context benchmarks using `mweralign`. You can download the sentencepiece model directly using wget:

```
wget https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/sentencepiece.bpe.model
```


```bash
export NLLB_TOKENIZER=''

# Run Short context evaluations
bash run_evals_mcif_short.sh

# Run Long context evaluations (Requires NLLB_TOKENIZER)
bash run_evals_mcif_long.sh
```
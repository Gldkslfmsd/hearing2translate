import os
import json
import random
import shutil

os.makedirs("evaluation_human/hearing2translate-v1/", exist_ok=True)

DOCS_PER_TASK = 60
USERS = {
    "en-it": "enit",
    "en-es": "enes",
    "en-de": "ende",
    "en-pt": "enpt",
    "en-zh": "enzh",
    "en-fr": "enfr",
    "en-nl": "ennl",
    "de-en": "deen",
    "es-en": "esen",
    "it-en": "iten",
    "pt-en": "pten",
    "zh-en": "zhen",
}
TASKS_PER_LANG = 3
MODELS = ["seamlessm4t", "aya_canary-v2", "voxtral-small-24b"]
DATASET = "covost2"
tasks_users = []

os.makedirs(f"evaluation_human/hearing2translate-v1/assets/{DATASET}", exist_ok=True)

for langs in USERS.keys():
    r_local = random.Random(0)

    with open(f"manifests/{DATASET}/{langs}.jsonl", "r") as f:
        data_src_raw = [json.loads(line) for line in f]
        data_src = data_src_raw[:DOCS_PER_TASK*TASKS_PER_LANG]

    data_tgt = {}
    for model in MODELS:
        with open(f"outputs/{model}/{DATASET}/{langs}.jsonl", "r") as f:
            data_tgt[model] = [
                json.loads(line)
                for line in f
            ]
            data_tgt[model] = data_tgt[model][:DOCS_PER_TASK*TASKS_PER_LANG]

            assert all([x["sample_id"] == y["sample_id"] for x, y in zip(data_src, data_tgt[model])]), "Sample IDs do not match!"

    # transpose
    data_tgt = [
        { 
            model: data_tgt[model][i]
            for model in MODELS
        }
        for i in range(len(data_src))
    ]

    tasks_users.append([])
    
    for src, tgts in zip(data_src, data_tgt):
        tgts = list(tgts.items())
        r_local.shuffle(tgts)
        
        lang1 = src['src_audio'].split('/')[-2]
        fname0 = f"manifests/covost2/audio/covost_{lang1}/{lang1}/{src['src_audio'].split('/')[-1]}"
        fname1 = f"evaluation_human/hearing2translate-v1/assets/{DATASET}/{lang1}/{src['src_audio'].split('/')[-1]}"

        if not os.path.exists(fname1):
            os.makedirs(os.path.dirname(fname1), exist_ok=True)
            shutil.copy(fname0, fname1)

        # each item is a document
        tasks_users[-1].append([{
            "langs": langs,
            "dataset": DATASET,
            "sample_id": src["sample_id"],
            "models": [v[0] for v in tgts],
            "src": f'<audio controls="" src="assets/{DATASET}/{lang1}/{src['src_audio'].split('/')[-1]}"></audio>',
            "tgt": [
                v[1]["output"]
                for v in tgts
            ]
        }])

        if len(tasks_users[-1]) == DOCS_PER_TASK:
            tasks_users.append([])

    # filter out empty task
    tasks_users = [task for task in tasks_users if task]

campaign = {
    "campaign_id": "hearing2translate-v1",
    "info": {
        "assets": {
            "source": f"hearing2translate-v1/assets/{DATASET}/",
            "destination": f"assets/{DATASET}/"
        },
        "template": "listwise",
        "assignment": "task-based",
        "protocol_score": True,
        "protocol_error_spans": True,
        "protocol_error_categories": True,
        "users": [
            user
            for langs in USERS.keys()
            for user in [f"{USERS[langs]}{i+1}" for i in range(TASKS_PER_LANG)]
        ]
    },
    "data": tasks_users,
}

with open("evaluation_human/hearing2translate-v1/campaign.json", "w") as f:
    json.dump(campaign, f, indent=2, ensure_ascii=False)
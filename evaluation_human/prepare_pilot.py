import os
import json
import random
import shutil

os.makedirs("evaluation_human/data_campaign", exist_ok=True)

# each "document" is 10 segments
# we want to have 5 documents
# and 2 users per language pair
N_PER_DOC = 10
N_DOCS = 5
USERS = {
    "en-it": "sara",
    "en-es": "javi",
    "en-de": "maikele",
}
TASKS_PER_LANG = 2
MODELS = ["seamlessm4t", "aya_whisper", "spirelm"]
DATASET = "mexpresso"
tasks_users = []

os.makedirs(f"evaluation_human/data_campaign/assets/{DATASET}", exist_ok=True)

for langs in USERS.keys():
    r_local = random.Random(0)

    with open(f"manifests/{DATASET}/{langs}.jsonl", "r") as f:
        data_src_raw = [json.loads(line) for line in f]
        data_src = []
        # sample continuous ranges
        for _ in range(N_DOCS*TASKS_PER_LANG):
            i = r_local.choice(range(len(data_src_raw)-N_PER_DOC))
            for _ in range(N_PER_DOC):
                data_src.append(data_src_raw.pop(i))

    data_tgt = {}
    for model in MODELS:
        with open(f"outputs/{model}/{DATASET}/{langs}.jsonl", "r") as f:
            data_tgt[model] = [
                json.loads(line)
                for line in f
            ]
            data_tgt[model] = [
                [item for item in data_tgt[model] if item["sample_id"] == src["sample_id"]][0]
                for src in data_src
            ]

    # transpose
    data_tgt = [
        {
            model: data_tgt[model][i]
            for model in MODELS
        }
        for i in range(len(data_src))
    ]

    tasks = []
    docs = []
    
    for src, tgts in zip(data_src, data_tgt):
        tgts = list(tgts.items())
        r_local.shuffle(tgts)

        fname0 = f"evaluation_human/data_campaign/mexpresso/{src['src_audio'].split('/')[-1]}"
        fname1 = f"evaluation_human/data_campaign/assets/{DATASET}/{src['src_audio'].split('/')[-1]}"

        if not os.path.exists(fname1):
            shutil.copy(fname0, fname1)


        # TODO: move audio to assets
        docs.append({
            "langs": langs,
            "sample_id": f"{DATASET}_{src["sample_id"]}",
            "models": [v[0] for v in tgts],
            "src": f'<audio controls="" src="assets/{DATASET}/{src['src_audio'].split('/')[-1]}"></audio>',
            "tgt": [
                v[1]["output"]
                for v in tgts
            ]
        })

        if len(docs) == N_PER_DOC:
            tasks.append(docs)
            docs = []
    if len(docs) > 0:
        tasks.append(docs)

    tasks_users += [
        tasks[i*N_DOCS:(i+1)*N_DOCS]
        for i in range(TASKS_PER_LANG)
    ]


campaign = {
    "campaign_id": "pilot",
    "info": {
        "assets": f"data_campaign/{DATASET}",
        "template": "listwise",
        "assignment": "task-based",
        "protocol_score": True,
        "protocol_error_spans": True,
        "protocol_error_categories": True,
        # two for italian, three for spanish, two for german
        "users": [
            user
            for langs in USERS.keys()
            for user in [f"{USERS[langs]}{i+1}" for i in range(TASKS_PER_LANG)]
        ]
    },
    "data": tasks_users,
}

with open("evaluation_human/data_campaign/pilot.json", "w") as f:
    json.dump(campaign, f, indent=2, ensure_ascii=False)
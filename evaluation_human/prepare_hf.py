# %%

import json
from datasets import Dataset, Audio

with open("hearing2translate-v1/annotations_metrics.json", "r") as f:
    data_metrics = json.load(f)
    for line in data_metrics:
        line.pop("human_score")
        line["xcomet_qe_by_100"] = line.pop("xcomet_qe_score_strict")
    data_metrics = {
        (x.pop("langs"), x.pop("system"), x.pop("sample_id")): x
        for x in data_metrics
    }

# human_score
# xcomet_qe_score_strict
# metricx_qe_score_strict
# linguapy_score
# metricx_qe_normalized
# xcomet_qe_by_100

with open("hearing2translate-v1/annotations.json", "r") as f:
    data = json.load(f)["hearing2translate-v1"]

LANGS = {x["item"][0]["langs"] for x in data}
data_langs = {}
for langs in LANGS:
    with open(f"../manifests/covost2/{langs}.jsonl", "r") as f:
        data_langs_raw = [json.loads(line) for line in f]
        for line in data_langs_raw:
            data_langs[(langs, line["sample_id"])] = line

for line in data:
    item = line.pop("item")
    assert len(item) == 1
    item = item[0]
    line.pop("user_id")
    line.pop("item_i")
    line["sample_id"] = item["sample_id"]
    line["dataset"] = item["dataset"]
    # old version of Pearmut doesnt have auto model mapping
    line["tgt"] = {
        model: tgt
        for model, tgt in zip(item["models"], item["tgt"])
    }
    line["langs"] = item["langs"]

    line["src_ref"] = data_langs[(item["langs"], item["sample_id"])]["src_ref"]
    line["tgt"]["ref"] = data_langs[(item["langs"], item["sample_id"])]["tgt_ref"]
    line["src_audio"] = data_langs[(item["langs"], item["sample_id"])]["src_audio"].removeprefix("/covost/audio/")

    for action in line["actions"]:
        if "candidate" in action:
            action["model"] = item["models"][action.pop("candidate")]

    line["annotations"] = {
        model: annotation
        for model, annotation in zip(item["models"], line.pop("annotations")[0])
    }

    line["metrics"] = {
        model: 
        data_metrics.get((line["langs"], model, line["sample_id"]), {})
        for model in item["models"]
    }

# change order of keys
data = [
    {
        "src_audio": x["src_audio"],
        "src_ref": x["src_ref"],
        "tgt": x["tgt"],
        "annotations": x["annotations"],
        "metrics": x["metrics"],
        "actions": x["actions"],
        "dataset": x["dataset"],
        "langs": x["langs"],
        "sample_id": x["sample_id"],
    }
    for x in data
]

# mark as train split
dataset = Dataset.from_list(data, split="train")
dataset.push_to_hub("zouharvi/hearing2translate-humeval")
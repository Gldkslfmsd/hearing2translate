# %%

import json

with open("generated/hearing2translate-v1-annotations.json", "r") as f:
    data = json.load(f)["hearing2translate-v1"]

# %%
import collections
import numpy as np

data_per_user = collections.defaultdict(list)
for line in data:
    data_per_user[line["user_id"]].append(line)

ranks_all_all = []
for lang, items in data_per_user.items():
    ranks_all = []
    for item in items:
        # NOTE: take the first item in a document since our documents are single-item
        scores = [x["score"] for x in item["annotations"][0]]
        # Sort indices by score (descending)
        sorted_indices = sorted(
            range(len(scores)), key=lambda k: scores[k], reverse=True
        )
        ranks = [0] * len(scores)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank
        # Handle ties: assign the lowest rank to all tied scores
        score_to_rank = {}
        for idx in sorted_indices:
            score = scores[idx]
            if score not in score_to_rank:
                score_to_rank[score] = ranks[idx]
        for idx in range(len(scores)):
            ranks[idx] = score_to_rank[scores[idx]]

        ranks_all.append(ranks)
        ranks_all_all.append(ranks)

    print(
        f"User {lang}: mean ranks per position: {np.mean(ranks_all, axis=0).round(2)}"
    )

print(f"Average ranks per position: {np.mean(ranks_all_all, axis=0).round(2)}")


# %%

model_scores = collections.defaultdict(lambda: collections.defaultdict(list))
for lang, items in data_per_user.items():
    if lang == "zhen1":
        continue
    for item in items:
        for model, annotation in zip(item["item"][0]["models"], item["annotations"][0]):
            model_scores[model][lang].append(annotation["score"])

langs_all = ["ende1", "enes1", "enit1", "enzh1", "ennl1", "deen1", "esen1", "iten1"]

MODEL_TO_NAME = {
    "seamlessm4t": r"\cellcolor{sfmcolor} \hspace{-1mm}\seamlessfixed",
    "aya_canary-v2":  r"\cellcolor{cascadecolor} \canary + \aya",
    "voxtral-small-24b": r"\cellcolor{speechllmcolor}{\voxtral}",
}

def get_cell(value):
    color = "GenericColor"
    minv, maxv = 60, 100
    color_v = ((value - minv) / (maxv - minv)) * 100
    color_v = max(0, min(100, color_v))
    return f"\\cellcolor{{{color}!{color_v:.0f}}} {value:.2f}"

models = sorted(
    model_scores.keys(),
    key=lambda x: np.mean([score for lang in langs_all for score in model_scores[x][lang]]
), reverse=True)

with open("generated/humeval-scores.tex", "w") as f:
    print(r"\begin{tabular}{r" + "m{1.0cm}" * len(models) + "}", file=f)
    print(r"\toprule", file=f)
    print(
        "",
        *[
            MODEL_TO_NAME.get(model, model) for model in models
        ],
        end=" \\\\\n",
        sep=" & ",
        file=f,
    )
    print(r"\midrule", file=f)
    avg_per_model = [
        f"{np.mean([score for l in langs_all for score in model_scores[model][l]]):.2f}"
        for model in models
    ]
    print(
        "Average",
        *[get_cell(float(x)) for x in avg_per_model],
        end=" \\\\[-0.5em]\\\\",
        sep=" & ",
        file=f,
    )
    for lang in langs_all:
        lang1, lang2 = lang[:2], lang[2:4]
        print(
            f"{lang1}-{lang2}",
            *[
                get_cell(np.mean(model_scores[model][lang]))
                for model in models
            ],
            end=" \\\\\n",
            sep=" & ",
            file=f,
        )

            
    print(r"\bottomrule", file=f)
    print(r"\end{tabular}", file=f)


# %%

# analyze error categories

model_errors = collections.defaultdict(list)
for lang, items in data_per_user.items():
    for item in items:
        for model, annotation in zip(item["item"][0]["models"], item["annotations"][0]):
            model_errors[model] += [
                error["category"]
                for error in annotation["error_spans"]
            ]

model_errors_all = collections.Counter(
    error
    for errors in model_errors.values()
    for error in errors
)
model_errors = {
    model: collections.Counter(errors)
    for model, errors in model_errors.items()
}
errors_total = {model: sum(errors.values()) for model, errors in model_errors.items()}

with open("generated/humeval-errors.tex", "w") as f:
    print(r"\begin{tabular}{@{}r@{\hspace{2mm}}" + "m{1.0cm}" * len(models) + "}", file=f)
    print(r"\toprule", file=f)
    print(
        "",
        *[
            MODEL_TO_NAME.get(model, model)  + r"\hspace{-4mm}" for model in models
        ],
        end=" \\\\\n",
        sep=" & ",
        file=f,
    )
    print(r"\midrule", file=f)
    for error_type, _ in model_errors_all.most_common():
        print(
            (
                error_type
                .replace("Inconsistent use of terminology", "Inconsistent")
                .replace("Linguistic conventions", "Linguistic")
                .replace("_", " ")
                .title()
            ),
            *[
                f"{model_errors[model][error_type]} \\tiny {(model_errors[model][error_type] / errors_total[model] * 100):.1f}\\%"
                for model in models
            ],
            end=" \\\\\n",
            sep=" & ",
            file=f,
        )
    print(r"\bottomrule", file=f)
    print(r"\end{tabular}", file=f)
# %%

import json

with open("hearing2translate-v1/annotations.json", "r") as f:
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
    "aya_canary-v2": r"\cellcolor{cascadecolor} \canary + \aya",
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
    key=lambda x: np.mean(
        [score for lang in langs_all for score in model_scores[x][lang]]
    ),
    reverse=True,
)

with open("generated/humeval-scores.tex", "w") as f:
    print(r"\begin{tabular}{r" + "m{1.0cm}" * len(models) + "}", file=f)
    print(r"\toprule", file=f)
    print(
        "",
        *[MODEL_TO_NAME.get(model, model) for model in models],
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
            *[get_cell(np.mean(model_scores[model][lang])) for model in models],
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
                error["category"] for error in annotation["error_spans"]
            ]

model_errors_all = collections.Counter(
    error for errors in model_errors.values() for error in errors
)
model_errors = {
    model: collections.Counter(errors) for model, errors in model_errors.items()
}
errors_total = {model: sum(errors.values()) for model, errors in model_errors.items()}

with open("generated/humeval-errors.tex", "w") as f:
    print(
        r"\begin{tabular}{@{}r@{\hspace{2mm}}" + "m{1.0cm}" * len(models) + "}", file=f
    )
    print(r"\toprule", file=f)
    print(
        "",
        *[MODEL_TO_NAME.get(model, model) + r"\hspace{-4mm}" for model in models],
        end=" \\\\\n",
        sep=" & ",
        file=f,
    )
    print(r"\midrule", file=f)
    for error_type, _ in model_errors_all.most_common():
        print(
            (
                error_type.replace("Inconsistent use of terminology", "Inconsistent")
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


# %%

# analyze correlations with metrics

import scipy.stats
import json
import collections
import statistics
import numpy as np

with open("hearing2translate-v1/annotations_metrics.json", "r") as f:
    data = json.load(f)


def is_constant(xs):
    return all(x == xs[0] for x in xs)


def global_pearson(data):
    if is_constant([human_score for score, human_score, system, sample_id in data]):
        return 0.0
    return scipy.stats.pearsonr(
        [score for score, human_score, system, sample_id in data],
        [human_score for score, human_score, system, sample_id in data],
    ).correlation


def global_spearman(data):
    if is_constant([human_score for score, human_score, system, sample_id in data]):
        return 0.0
    return scipy.stats.spearmanr(
        [score for score, human_score, system, sample_id in data],
        [human_score for score, human_score, system, sample_id in data],
    ).correlation


def byitem_spearman(data):
    agg_item = collections.defaultdict(list)
    for score, human_score, system, sample_id in data:
        agg_item[sample_id].append((score, human_score))

    corrs = [
        (
            0
            if is_constant([y for x, y in v])
            else (
                1
                if is_constant([x for x, y in v])
                else scipy.stats.spearmanr(
                    [x for x, y in v],
                    [y for x, y in v],
                ).correlation
            )
        )
        for sample_id, v in agg_item.items()
    ]
    return statistics.mean([x if not np.isnan(x) else 0 for x in corrs])


results_byitem_spearman = collections.defaultdict(dict)
results_global_pearson = collections.defaultdict(dict)
results_global_spearman = collections.defaultdict(dict)

for lang in {x["langs"] for x in data}:
    print(f"Language pair: {lang}")
    for metric in ["xcomet_qe_score_strict", "metricx_qe_score_strict"]:
        data_xy = [
            (float(x[metric]), float(x["human_score"]), x["system"], x["sample_id"])
            for x in data
            if x["langs"] == lang
        ]
        if metric == "metricx_qe_score_strict":
            # invert scores for MetricX strict
            data_xy = [(-x, y, s, i) for x, y, s, i in data_xy]
        results_byitem_spearman[metric][lang] = byitem_spearman(data_xy)
        results_global_pearson[metric][lang] = global_pearson(data_xy)
        results_global_spearman[metric][lang] = global_spearman(data_xy)

results_all = [
    results_global_pearson,
    results_byitem_spearman,
    # results_global_spearman,
]


def get_cell1(value):
    color = "GenericColor"
    minv, maxv = 0, 0.2
    color_v = ((value - minv) / (maxv - minv)) * 100
    color_v = max(0, min(100, color_v))
    prefix = r"\phantom{-}" if value >= 0 else ""
    return f"\\cellcolor{{{color}!{color_v:.0f}}} {prefix}{value:.3f}"


def get_cell2(value):
    color = "GenericColor"
    minv, maxv = 0.2, 0.6
    color_v = ((value - minv) / (maxv - minv)) * 100
    color_v = max(0, min(100, color_v))
    return f"\\cellcolor{{{color}!{color_v:.0f}}} \\phantom{{-}}{value:.3f}"


with open("generated/humeval-correlations.tex", "w") as f:
    print(r"\begin{tabular}{r" + "p{0.9cm}" * 4 + "}", file=f)
    print(r"\toprule", file=f)
    print(
        "",
        *[
            f"\\multicolumn{{2}}{{c}}{{{metric}}}"
            for metric in [r"\cometstrict", r"\metricxstrict"]
        ],
        end=" \\\\\n",
        sep=" & ",
        file=f,
    )
    print(
        "",
        *[
            corr_type
            for metric in [r"\cometstrict", r"\metricxstrict"]
            for corr_type in ["global", "item"]
        ],
        end=" \\\\\n",
        sep=" & ",
        file=f,
    )
    print(r"\midrule", file=f)

    print(
        f"Average",
        *[
            (
                get_cell2(
                    statistics.mean(
                        [results[metric][lang] for lang in results[metric].keys()]
                    )
                )
                if results_i == 0
                else get_cell1(
                    statistics.mean(
                        [results[metric][lang] for lang in results[metric].keys()]
                    )
                )
            )
            for metric in ["xcomet_qe_score_strict", "metricx_qe_score_strict"]
            for results_i, results in enumerate(results_all)
        ],
        end=" \\\\\n",
        sep=" & ",
        file=f,
    )
    print(r"\\[-0.3em]", file=f)

    for lang in [
        "en-de",
        "en-es",
        "en-it",
        "en-zh",
        "en-nl",
        "de-en",
        "es-en",
        "it-en",
    ]:
        print(
            f"{lang}",
            *[
                (
                    get_cell2(results[metric][lang])
                    if results_i == 0
                    else get_cell1(results[metric][lang])
                )
                for metric in ["xcomet_qe_score_strict", "metricx_qe_score_strict"]
                for results_i, results in enumerate(results_all)
            ],
            end=" \\\\\n",
            sep=" & ",
            file=f,
        )
    print(r"\bottomrule", file=f)
    print(r"\end{tabular}", file=f)

"""
Combines multiple CSV files together into a single file (grouped by metrics) that can be used in GSheet.
Optionally also outputs a rendered LaTeX table.

Example usage: ```
python3 analysis/wmt/combine_csv.py \
    -i analysis/wmt/*.csv \
    -oc /home/vilda/Downloads/wmt_combined.csv \
    -ot /home/vilda/Downloads/wmt_combined.tex \
;
```
"""

import argparse
import csv
import collections

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", type=str, nargs="+",
                  required=True, help="Input CSV files")
args.add_argument("-oc", "--output-csv", type=str,
                  required=True, help="Output CSV file")
args.add_argument("-ot", "--output-tex", type=str,
                  required=False, help="Output TEX file")
args = args.parse_args()

data = collections.defaultdict(lambda: collections.defaultdict(dict))
metrics = None

for fname in args.input:
    langs = fname.split("/")[-1].split(".")[0].split("_", 1)[1].replace("_", "-")
    langs = "-".join(langs.split("-")[-2:])

    with open(fname, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:

            system = row.pop("system").replace("_", " ")

            values = {}
            for k, v in row.items():
                if v == "":
                    values[k] = "-"
                else:
                    values[k] = float(v)

            data[langs][system] = values
            if metrics is None:
                metrics = list(row.keys())


langs = [
    x for x in [
        'en-de',
        'en-es',
        'en-fr',
        'en-it',
        'en-nl',
        'en-pt',
        'en-zh',
        'de-en',
        'es-en',
        'fr-en',
        'it-en',
        'pt-en',
        'zh-en'
    ]
    if x in data.keys()
]


with open(args.output_csv, "w") as f:
    def printcsv(*args):
        print(
            *args,
            sep=",",
            file=f,
            end="\n",
        )

    printcsv(
        "system",
        *[
            x
            for metric in metrics
            for x in [metric] + ["" for _lang in langs[:-1]]
        ]
    )
    printcsv(
        "",
        *[
            lang
            for _ in metrics
            for lang in langs
        ]
    )
    for system in data[langs[0]].keys():
        printcsv(
            system,
            *[
                data[lang][system][metric]
                for metric in metrics
                for lang in langs
            ]
        )

METRIC_TO_NAME = {
    "LinguaPy": "LinguaPy",
    "QEMetricX_24-Strict-linguapy": r"\metricxstrictiny",
    "XCOMET-QE-Strict-linguapy": r"\cometstrictiny",
}
metrics = METRIC_TO_NAME.keys()

SYSTEM_TO_NAME = {
    "whisper": r"\cellcolor{sfmcolor} \whisper",
    "seamlessm4t": r"\cellcolor{sfmcolor} \seamless",
    "canary-v2": r"\cellcolor{sfmcolor} \canary",
    "owsm4.0-ctc": r"\cellcolor{sfmcolor} \owsm",

    "aya whisper": r"\cellcolor{cascadecolor} \whisperfixed \,+ \aya",
    "gemma whisper": r"\cellcolor{cascadecolor} \nonefixed \,+ \gemma",
    "tower whisper": r"\cellcolor{cascadecolor} \nonefixed \,+ \tower",
    "aya seamlessm4t": r"\cellcolor{cascadecolor} \seamlessfixed \,+ \aya",
    "gemma seamlessm4t": r"\cellcolor{cascadecolor} \nonefixed \,+ \gemma",
    "tower seamlessm4t":  r"\cellcolor{cascadecolor} \nonefixed \,+ \tower",
    "aya canary-v2":  r"\cellcolor{cascadecolor} \canaryfixed \,+ \aya",
    "gemma canary-v2": r"\cellcolor{cascadecolor} \nonefixed \,+ \gemma",
    "tower canary-v2": r"\cellcolor{cascadecolor} \nonefixed \,+ \tower",
    "aya owsm4.0-ctc": r"\cellcolor{cascadecolor} \owsmfixed \,+ \aya",
    "gemma owsm4.0-ctc": r"\cellcolor{cascadecolor} \nonefixed \,+ \gemma",
    "tower owsm4.0-ctc": r"\cellcolor{cascadecolor} \nonefixed \,+ \tower",

    "desta2-8b": r"\cellcolor{speechllmcolor}{\desta}",
    "qwen2audio-7b": r"\cellcolor{speechllmcolor}{\qwenaudio}",
    "phi4multimodal": r"\cellcolor{speechllmcolor}{\phimultimodal}",
    "voxtral-small-24b": r"\cellcolor{speechllmcolor}{\voxtral}",
    "spirelm": r"\cellcolor{speechllmcolor}{\spire}",
}

if args.output_tex:
    with open(args.output_tex, "w") as f:
        def printtex(*args):
            print(
                *args,
                sep=" & ",
                file=f,
                end=" \\\\\n",
            )

        def color_cell(value, metric):
            if value  == "-":
                return "-"
            color = {
                "LinguaPy": "Brown3",
                "metricx_qe_score": "Chartreuse3",
                "QEMetricX_24-Strict-linguapy": "Chartreuse3",
                "xcomet_qe_score": "DarkSlateGray3",
                "XCOMET-QE-Strict-linguapy": "DarkSlateGray3",
            }

            s = f"{value:.1f}"
            if metric in {"LinguaPy"}:
                color = "Brown3"
                minv, maxv = 0, -20
            elif metric in {"metricx_qe_score", "QEMetricX_24-Strict-linguapy"}:
                color = "Chartreuse3"
                minv, maxv = 20, 80
            elif metric in {"xcomet_qe_score", "XCOMET-QE-Strict-linguapy"}:
                color = "DarkSlateGray3"
                minv, maxv = 20, 80
            color_v = ((value - minv) / (maxv - minv)) * 100
            color_v = max(0, min(100, color_v))

            return f"\\cellcolor{{{color}!{color_v:.0f}}} {s}"

        print(
            r"\begin{tabular}{l" + "r" * ((len(langs)+1) * len(metrics)) + "}",
            r"\toprule",
            file=f,
        )
        printtex(
            "",
            *[
                f"\\multicolumn{{{len(langs)+1}}}{{c}}{{\\bf {METRIC_TO_NAME[metric]}}}"
                for metric in metrics
            ]
        )
        printtex(
            "",
            *[
                lang
                for _ in metrics
                for lang in langs + [""]
            ]
        )
        print("\\midrule", file=f)

        system_order = [
            (v, k) for k, v in SYSTEM_TO_NAME.items()
            if k in data[langs[0]].keys()
        ]
        for system, system_k in system_order:

            printtex(
                system,
                *[
                    "" if lang == "" else
                    "-" if system_k == "canary-v2" and lang in {"en-zh", "zh-en"} else
                    "-" if system_k.endswith(" canary-v2") and lang in {"zh-en"} else
                    color_cell(data[lang][system_k][metric], metric)
                    for metric in metrics
                    for lang in langs + [""]
                ]
            )

        print(r"\bottomrule \end{tabular}", file=f)

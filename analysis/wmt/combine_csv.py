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
import statistics

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", type=str, nargs="+", required=True, help="Input CSV files")
args.add_argument("-oc", "--output-csv", type=str, required=True, help="Output CSV file")
args.add_argument("-ot", "--output-tex", type=str, required=False, help="Output TEX file")
args = args.parse_args()

data = collections.defaultdict(lambda: collections.defaultdict(dict))
metrics = None

for fname in args.input:
    langs = fname.split("/")[-1].split(".")[0].split("_", 1)[1].replace("_", "-")
    with open(fname, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            system = row.pop("system")
            data[langs][system] = {k: float(v) for k, v in row.items()}
            
            if metrics is None:
                metrics = list(row.keys())

langs = list(data.keys())


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
    "QEMetricX_24-Strict-linguapy": "MetricX$^L$",
    "XCOMET-QE-Strict-linguapy": "XCOMET$^L$",
}
metrics = METRIC_TO_NAME.keys()

SYSTEM_ORDER = [
    "whisper",
    "seamlessm4t",
    "canary-v2",
    "owsm4.0-ctc",

    "aya whisper",
    "gemma whisper",
    "tower whisper",

    "aya seamlessm4t",
    "gemma seamlessm4t",
    "tower seamlessm4t",

    "aya canary-v2",
    "gemma canary-v2",
    "tower canary-v2",

    "aya owsm4.0-ctc",
    "gemma owsm4.0-ctc",
    "tower owsm4.0-ctc",

    "desta2-8b",
    "qwen2audio-7b",
    "phi4multimodal",
    "voxtral-small-24b",
    "spirelm",
]

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
            color_v = ( (value - minv) / (maxv - minv) ) * 100
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
                f"\\multicolumn{{{len(langs)}}}{{c}}{{\\bf {METRIC_TO_NAME[metric]}}}"
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

        # invert and scale metrics
        for lang in langs:
            for metric in metrics:
                if metric in {"metricx_qe_score", "QEMetricX_24-Strict-linguapy", }:
                    for system in data[lang].keys():
                        data[lang][system][metric] = 100-4*data[lang][system][metric]
                elif metric in {"LinguaPy"}:
                    for system in data[lang].keys():
                        data[lang][system][metric] = -data[lang][system][metric]
                elif metric in {"xcomet_qe_score", "XCOMET-QE-Strict-linguapy"}:
                    for system in data[lang].keys():
                        data[lang][system][metric] = 100*data[lang][system][metric]

        system_order = sorted(
            data[langs[0]].keys(),
            key=lambda k: SYSTEM_ORDER.index(k.replace("_", " ")),
        )
        for system in system_order:
            printtex(
                system.replace("_", r" "),
                *[
                    color_cell(data[lang][system][metric], metric) if lang != "" else ""
                    for metric in metrics
                    for lang in langs + [""]
                ]
            )

        print(r"\bottomrule \end{tabular}", file=f)
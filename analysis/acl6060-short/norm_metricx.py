"""
Normalize MetricX Score by 100 - 4 * score, and measure diff of {short - long}
"""

import pandas as pd

SYSTEMS = [
    "canary-v2",
    "owsm4.0-ctc",
    "aya_whisper",
    "gemma_whisper",
    "tower_whisper",
    "aya_canary-v2",
    "gemma_canary-v2",
    "tower_canary-v2",
    "aya_owsm4.0-ctc",
    "gemma_owsm4.0-ctc",
    "tower_owsm4.0-ctc",
    "desta2-8b",
    "qwen2audio-7b",
    "phi4multimodal",
    "voxtral-small-24b"
]
#METRICX = "QEMetricX_24-Strict-linguapy"
METRICX = "metricx_qe_score"

def main(langs, dataset="acl6060"):
    out_df = pd.DataFrame({"system": SYSTEMS})
    
    norm_fn = lambda x: 100 - 4 * x
    
    for lang in langs:
        # prepare files
        short_dir = f"{dataset}_{lang}.csv"
        long_dir = f"../{dataset}-long/{short_dir}"
        
        df_short = pd.read_csv(short_dir)
        df_short["short"] = df_short[METRICX].apply(norm_fn)
        
        df_long = pd.read_csv(long_dir)
        df_long["long"] = df_long[METRICX].apply(norm_fn)
        
        # compute diff
        short = df_short[["system", "short"]]
        long = df_long[["system", "long"]]
        df = pd.merge(short, long, on="system", how="inner")
        df["diff"] = df.short - df.long
        
        # sort by systems
        df["system"] = pd.Categorical(df["system"], categories=SYSTEMS, ordered=True)
        df = df.sort_values("system").reset_index(drop=True)
        
        df_lang = df[["system", "diff"]].rename(columns={"diff": lang})
        
        out_df = out_df.merge(df_lang, on="system", how="left")
        
    out_df["avg"] = out_df.iloc[:, 1:].mean(axis=1)
    out_df.to_csv("metricx_qe_diff.csv", index=False)
    print("done")
    
if __name__ == "__main__":
    langs = ["en_de", "en_fr", "en_pt", "en_zh"]
    main(langs)
"""
For ACL 60/60 and MCIF
"""
#!/bin/bash

save_dir=".."

tasks=(
  "analysis/acl6060-short/combine_csv.py analysis/acl6060-short/diff_*.csv  acl6060_diff_combined.csv  acl6060_diff_combined.tex  'ACL 6060 diff saved!'"
  "analysis/acl6060-short/combine_csv.py analysis/acl6060-short/acl6060_*.csv  acl6060_combined.csv  acl6060_combined.tex  'ACL 6060 absolute scores saved!'"
  "analysis/mcif-short/combine_csv.py analysis/mcif-short/diff_*.csv  mcif_diff_combined.csv  mcif_diff_combined.tex  'MCIF diff saved!'"
  "analysis/mcif-short/combine_csv.py analysis/mcif-short/mcif_*.csv  mcif_combined.csv  mcif_combined.tex  'MCIF absolute scores saved!'"
)

for task in "${tasks[@]}"; do
    read -r script input_glob out_csv out_tex message <<< "$task"

    files=( $input_glob )

    if (( ${#files[@]} == 0 )); then
        echo "[SKIP] No files matched glob: $input_glob"
        continue
    fi

    python3 "$script" \
        -i "${files[@]}" \
        -oc "$save_dir/$out_csv" \
        -ot "$save_dir/$out_tex"

    if [[ $? -eq 0 ]]; then
        echo "$message"
    fi
done
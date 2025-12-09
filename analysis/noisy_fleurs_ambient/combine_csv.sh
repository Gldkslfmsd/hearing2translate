BASE_PATH="/net/tscratch/people/plgzuefle/noisy_speech"

python3 $BASE_PATH/hearing2translate/analysis/noisy_fleurs_babble/combine_csv.py \
    -i $BASE_PATH/hearing2translate/analysis/noisy_fleurs_ambient/noisy_fleurs_*.csv \
    -oc $BASE_PATH/out_tables/noisy_ambient_combined.csv\
    -ot $BASE_PATH/out_tables/noisy_ambient_combined.tex \
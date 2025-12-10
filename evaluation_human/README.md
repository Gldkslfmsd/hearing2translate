This directory contains data wrangling scripts for a small-scale human evaluation.


To run the human interface, run:
```bash
pip install "pearmut>=0.2.5"
# H2T_DATADIR="evaluation_human/data_campaign/" python3 manifests/mexpresso/generate.py
# NOTE: manually move files to manifests/covost2/audio
python3 evaluation_human/prepare_v1.py
mv ./evaluation_human/hearing2translate-v1/ ~/pearmut/data_vm/hearing2translate-v1/

# this will create data/ directory that tracks progress
# run from ~ or some similar place ideally

cd evaluation_human
# use older version of pearmut with legacy data format
pip install "pearmut==0.2.11"
pearmut add hearing2translate-v1/hearing2translate-v1.json
pearmut run
```
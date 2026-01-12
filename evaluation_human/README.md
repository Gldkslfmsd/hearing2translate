This directory contains data wrangling scripts for a small-scale human evaluation.


To run the human interface, run:
```bash
pip install "pearmut>=1.0.0"
# H2T_DATADIR="evaluation_human/data_campaign/" python3 manifests/covost2/generate.py
# and manually move files to manifests/covost2/audio
# NOTE: data is not publicly available anymore, download from
# https://github.com/sarapapi/hearing2translate/releases/tag/data-share-covost2
python3 evaluation_human/prepare_v1.py

# this will create data/ directory that tracks progress
# run from ~ or some similar place ideally

cd evaluation_human
# add our campaign
pearmut add hearing2translate-v1/campaign.json
# launch server
pearmut run
```

The results with audios are compiled here online: [huggingface.co/datasets/zouharvi/hearing2translate-humeval](https://huggingface.co/datasets/zouharvi/hearing2translate-humeval).
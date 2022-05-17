#! /bin/sh
python experiments/glue/glue_san_prepro.py
python prepro_std.py --model bert-base-uncased --root_dir data/canonical_san_data --task_def experiments/glue/glue_san_task_def.yml --do_lower_case $1

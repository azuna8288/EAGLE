# cd eagle/ge_data 
# python3 allocation.py --outdir ../../data
# cd ../../


accelerate launch -m --main_process_port 1234 --mixed_precision bf16 eagle.train.main --basepath /opt/tiger/mariana/EAGLE/EAGLE_files/241114_3b3_sft30_12b-kd-bo128_hf --configpath /opt/tiger/mariana/EAGLE/eagle/train/p6_draft_config.json --tmpdir data/  --cpdir EAGLE_files/output/

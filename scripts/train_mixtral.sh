cd eagle/ge_data 
python3 allocation_mixtral.py --outdir ../../data

cd ../../
accelerate launch -m --main_process_port 1234 --mixed_precision bf16 eagle.train.main --basepath /opt/tiger/mariana/EAGLE/hub/Mixtral-8x7B-Instruct-v0.1 --configpath /opt/tiger/mariana/EAGLE/eagle/train/Mixtral_8x7B_config.json --tmpdir data/  --cpdir output/

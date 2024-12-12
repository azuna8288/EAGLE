export CUDA_VISIBLE_DEVICES=0
python3 -m eagle.evaluation.gen_ea_alpha_p6 \
		 --ea-model-path /opt/tiger/mariana/EAGLE/241114_draft_eagle \
		 --base-model-path /opt/tiger/mariana/EAGLE/241114_3b3_sft30_12b-kd-bo128_hf \


# python3 eagle/evaluation/alpha.py
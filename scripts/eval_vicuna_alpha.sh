export CUDA_VISIBLE_DEVICES=0
python3 -m eagle.evaluation.gen_ea_alpha_vicuna \
		 --ea-model-path /opt/tiger/mariana/EAGLE/hub/EAGLE-Vicuna-13B-v1.3 \
		 --base-model-path /opt/tiger/mariana/EAGLE/hub/vicuna-13b-v1.3 \
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m eagle.evaluation.gen_ea_alpha_llama2chat \
		 --ea-model-path /opt/tiger/mariana/EAGLE/EAGLE_files/hub/EAGLE-mixtral-instruct-8x7B \
		 --base-model-path /opt/tiger/mariana/EAGLE/EAGLE_files/hub/Mixtral-8x7B-Instruct-v0.1

# python3 -m eagle.evaluation.alpha
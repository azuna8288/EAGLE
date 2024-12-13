export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd eagle/
python3 -m eagle.evaluation.gen_ea_alpha_llama2chat \
		 --ea-model-path /opt/tiger/mariana/EAGLE/EAGLE_files/output/for_debug \
		 --base-model-path /opt/tiger/mariana/EAGLE/EAGLE_files/241114_3b3_sft30_12b-kd-bo128_hf \
		 --tree-choices chain

# python3 -m eagle.evaluation.alpha
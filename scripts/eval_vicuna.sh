export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd eagle/
python3 -m eagle.evaluation.gen_ea_alpha_vicuna \
		 --ea-model-path $HOME_DIR/output/latest \
		 --base-model-path $HOME_DIR/hub/vicuna-7b-v1.3 \
		 --tree-choices chain \
         --temperature 1.0 \
         --question-begin 0 --question-end 2\
		 --answer-file $HOME_DIR/output/mt_bench_answer_temp1.jsonl
		 
python3 -m eagle.evaluation.alpha --anwer_file_path $HOME_DIR/output/mt_bench_answer_temp1.jsonl




python3 -m eagle.evaluation.gen_ea_alpha_vicuna \
		 --ea-model-path $HOME_DIR/output/latest \
		 --base-model-path $HOME_DIR/hub/vicuna-7b-v1.3 \
		 --tree-choices chain \
         --temperature 0.0 \
         --question-begin 0 --question-end 2\
		 --answer-file $HOME_DIR/output/mt_bench_answer_temp0.jsonl

python3 -m eagle.evaluation.alpha --anwer_file_path $HOME_DIR/output/mt_bench_answer_temp0.jsonl
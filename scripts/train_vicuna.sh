


accelerate launch -m --main_process_port 1234  --mixed_precision=bf16 eagle.train.main \
    --tmpdir data/ \
    --basepath hub/vicuna-7b-v1.3/ \
    --configpath eagle/train/vicuna_7B_config.json \
    --cpdir output/ 
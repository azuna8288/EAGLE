


accelerate launch -m --main_process_port 1234  --mixed_precision=bf16 eagle.train.main \
    --tmpdir data/ \
    --basepath hub/P61_D73_8B_official/ \
    --configpath draft_configs/P6_dense_8B_draft.json \
    --cpdir output/ 
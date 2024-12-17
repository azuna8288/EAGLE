
python3 -m eagle.ge_data.allocation --outdir data/

accelerate launch -m --main_process_port 1234  --mixed_precision=bf16 eagle.train.main \
    --tmpdir data/ \
    --basepath hub/3b3_moe_p6/ \
    --configpath draft_configs/P6_3B3_MoE_draft.json \
    --cpdir output/ 
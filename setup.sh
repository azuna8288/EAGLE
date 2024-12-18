pip3 install bytedance.seed_tokenizer
pip3 install -e .
pip3 install byted-seed-models==1.0.0
pip3 install modelscope
pip3 install datasets
pip3 install accelerate==0.26.0
pip3 install -U pyarrow


mkdir hub/
mkdir output/
mkdir data/

cd hub/
hdfs dfs -get hdfs://haruna/home/byte_data_seed/ssd_lq/public/seed_models/3b3_moe_p6/
cd ..
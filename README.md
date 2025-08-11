# How to build MSTS?

First follow these steps to build MSTS training dataset.
```bash
cd benchmark/redpajama
bash download.sh
python filter.py
python trauncate.py
```
Download checkpoints from [HuggingFace](https://huggingface.co/).
Take Qwen2.5-7B-Instruct as an example.
```bash
cd ./checkpoints
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
```
Convert the weights using convert_hf_checkpoint.py.
```bash
python convert_hf_checkpoint.py \
    --hf_repo_path ./checkpoints/Qwen2.5-7B-Instruct \
    --save_path ./checkpoints/Qwen2.5-7B-Instruct/convert \
    --weight_map_path ./checkpoints/weight_map/qwen2.5.json
```

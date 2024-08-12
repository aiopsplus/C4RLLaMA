# Supplementary Material

This package contains supplementary material for the paper "Code Comment Inconsistency Detection and Rectification Using a Large Language Model". The package is organized as follows:

- `Data.7z`: Contains the data used in our study.
- `templates/`: Contains the LLaMA templates used in our study.
- `utils/BalanceTrainer.py`: The loss function used in our study.
- `utils/prompter.py`: The prompter used in our study.
- `train.py`: The training script used in our study.
- `test.py`: The testing script used in our study.

run `train.py` to train the model.

```bash
python -u train.py --base_model codellama/CodeLlama-7b-hf \
--data_path Data/LLMtrainDataset.jsonl --output_dir ./LoraCodeLlama_7B --batch_size 32 --micro_batch_size 2 \
--num_epochs 10 --learning_rate 1e-4 --cutoff_len 2048 --val_set_size 100 --prompt_template_name llama \
--label_smoothing_factor 0.1 --classification_alpha 0.5 --train_on_inputs False
```

run `test.py` to test the model.

```bash
 python -u test.py --base_model codellama/CodeLlama-7b-hf --lora_weights ./LoraCodeLlama_7B --prompt_template llama
```
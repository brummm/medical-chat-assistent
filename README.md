# Medical Chat Assitent

This project aims to help healthcare workers diagnosing diseases through a LLM chatbot that uses the [MedQuAD project](https://github.com/abachaa/MedQuAD) as a dataset to fine tune a LLM model.

## Fine-tuning

### Acquire and prepare data

To prepare the data for fine-tuning, you can use the `fine-tune/acquire-data.sh` bash script. This script will:
1. Clone the MedQuAD repository into `./fine-tune/raw-data` (this directory is ignored by git).
2. Run the data transformation script `fine-tune/format-data.py` to generate MLX-ready `.jsonl` files in `./fine-tune/mlx-data/`.

Run the following command from the project root:

```bash
chmod +x fine-tune/acquire-data.sh
./fine-tune/acquire-data.sh
```

### Train

After running the commands from the section above, run the commands below:

```bash
python3 ./fine-tune/train.py
```

### Test and Compare

To test the fine-tuned model and compare it with the original base model, run:

```bash
python3 ./fine-tune/test.py
```
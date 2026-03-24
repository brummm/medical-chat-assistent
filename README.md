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

## Database Setup

Before running the main application, you need to initialize the SQLite database and populate it with example patient data and medical histories.

Run the following scripts from the project root using the virtual environment:

1. **Populate Patients:** Creates 5 example patients with diverse ages and sexes.
```bash
.venv/bin/python3 scripts/populate_patients.py
```
*Example Data:* Maria Silva (68yo), João Pereira (33yo), Ana Costa (10yo), Ricardo Santos (45yo), Sofia Oliveira (24yo).*

2. **Populate Visits:** Generates random, non-repeating medical visits for each patient over the last 2 years, including symptoms, diagnoses, and prescriptions based on the fine-tuning dataset.
```bash
.venv/bin/python3 scripts/populate_visits.py
```

## Running the Medical Assistant

The main application integrates the fine-tuned LLaMA-3 model with a LangChain pipeline, a ChromaDB Retrieval-Augmented Generation (RAG) system, and a local SQLite patient database.

To start the interactive medical assistant, run the following command from the project root:

```bash
.venv/bin/python3 main.py
```

### Capabilities

When running the main script, the assistant provides the following features:

- **Patient Context Integration:** Prompts you to select a patient from the local database. The assistant automatically retrieves and analyzes the patient's age, sex, and last 3 medical visits (symptoms, diagnoses, prescriptions) to provide highly contextualized answers.
- **Retrieval-Augmented Generation (RAG):** Uses ChromaDB to search through the entire MedQuAD dataset in real-time, fetching the most relevant medical evidence to ground its responses.
- **Source Citation:** Automatically appends the source URL of the medical information it used to formulate the answer.
- **Conversational Memory:** Maintains context throughout the chat session, allowing for coherent follow-up questions.
- **Strict Safety Guardrails:** Adheres to strict medical safety rules, preventing direct prescriptions and automatically appending a mandatory clinical validation disclaimer to every response.
- **Audit Logging:** Every interaction, including the user prompt, retrieved medical context, patient history, and model response, is securely logged to `./persisted/logs/clinical_interactions.jsonl` for hospital compliance and auditing.

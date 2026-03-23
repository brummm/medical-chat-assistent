import os
import sys

# Ensure project root is in path to import src/classes/model.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classes.model import MedicalModel
from src.classes.retriever import MedicalRetriever

def main():
    print(f"--- Comparison Test ---")
    
    # Initialize Retriever
    print("1. Initializing Retriever and Indexing...")
    retriever = MedicalRetriever()
    # Path to the training data for indexing
    retriever.build_index("./fine-tune/mlx-data/train.jsonl")

    # Original model uses default model_name but no adapters
    print("\n2. Loading Original Base Model...")
    original_model = MedicalModel(adapter_path=None)
    original_model.load_model()

    # Fine-tuned model uses all default values (Llama-3 + ./persisted adapters)
    print("\n3. Loading Fine-tuned Model...")
    ft_model = MedicalModel()
    ft_model.load_model()

    print("\n--- Medical Chatbot Comparison ---")
    print("Compare how the model behaves before and after fine-tuning (with RAG).")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_question = input("\nPatient Question: ")
        except EOFError:
            break
            
        if user_question.lower() in ["exit", "quit"]:
            break

        if not user_question.strip():
            continue

        # Retrieve Context (RAG)
        print("Retrieving context...", end="\r")
        context = retriever.retrieve(user_question)

        # Ask Original Model (No RAG)
        print("Generating Original response...", end="\r")
        original_response = original_model.ask(user_question)

        # Ask Fine-tuned Model (With RAG)
        print("Generating Fine-tuned + RAG response...", end="\r")
        ft_rag_response = ft_model.ask(user_question, context=context)
        
        print("\n" + "="*60)
        print(f"ORIGINAL MODEL RESPONSE (No RAG):")
        print(f"{original_response}")
        print("-" * 60)
        print(f"FINE-TUNED MODEL RESPONSE (With RAG):")
        print(f"{ft_rag_response}")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()

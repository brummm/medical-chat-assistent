import os
import sys
from datetime import datetime
from typing import Any, List, Optional

# LangChain imports
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# Project imports
from src.classes.model import MedicalModel
from src.classes.retriever import MedicalRetriever
from src.classes.database import MedicalDatabase
from src.classes.logger import MedicalAuditLogger

# Custom LLM Wrapper for MLX
class MLXLLM(LLM):
    medical_model: Any = None
    
    @property
    def _llm_type(self) -> str:
        return "mlx-model"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self.medical_model.ask(prompt)

def calculate_age(dob_str):
    dob = datetime.strptime(dob_str, "%Y-%m-%d")
    today = datetime.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

def get_patient_summary(db, patient_id):
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
        patient = cursor.fetchone()
        if not patient:
            return None
        
        history = db.get_patient_history(patient_id)
        last_visit = history[0] if history else None
        
        return {
            "info": dict(patient),
            "age": calculate_age(patient["date_of_birth"]),
            "last_visit": last_visit,
            "full_history": history
        }

def format_history_for_model(history, limit=3):
    if not history:
        return "No previous visits recorded."
    
    # history is already DESC from database
    recent_history = history[:limit]
    
    formatted = ""
    for visit in reversed(recent_history): # Show in chronological order for the model
        formatted += f"- Date: {visit['timestamp']}\n"
        formatted += f"  Symptoms: {visit['symptoms']}\n"
        formatted += f"  Diagnoses: {', '.join(visit['diagnoses'])}\n"
        formatted += f"  Prescriptions: {', '.join(visit['prescriptions'])}\n\n"
    
    if len(history) > limit:
        formatted = f"(Note: {len(history) - limit} older visits omitted for brevity)\n" + formatted
        
    return formatted

def format_chat_history(chat_history):
    formatted = ""
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            formatted += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            # Clean the disclaimer from history so it doesn't loop or clutter context
            clean_content = msg.content.replace("\n\n*** MANDATORY DISCLAIMER: Clinical validation by a physician is required before any medical action. ***", "")
            formatted += f"Assistant: {clean_content}\n\n"
    return formatted

def main():
    print("Initializing Medical Assistant...")
    db = MedicalDatabase()
    retriever = MedicalRetriever()
    audit_logger = MedicalAuditLogger()
    
    # Initialize and load model
    medical_model = MedicalModel()
    medical_model.load_model()
    
    # Wrap in LangChain
    llm = MLXLLM(medical_model=medical_model)

    while True:
        print("\n" + "="*50)
        print("PATIENT SELECTION")
        print("="*50)
        
        # Show all patients for convenience
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name FROM patients")
            patients = cursor.fetchall()
            for p in patients:
                print(f"[{p['id']}] {p['name']}")
        
        patient_input = input("\nEnter Patient ID (or 'exit' to quit): ").strip()
        if patient_input.lower() == 'exit':
            break
        
        try:
            patient_id = int(patient_input)
        except ValueError:
            print("Invalid ID.")
            continue
            
        summary = get_patient_summary(db, patient_id)
        if not summary:
            print("Patient not found.")
            continue
            
        # Display Patient Info
        info = summary['info']
        print("\n" + "-"*30)
        print(f"PATIENT: {info['name']}")
        print(f"Age: {summary['age']} | Sex: {info['sex']}")
        
        last = summary['last_visit']
        if last:
            print(f"Last Visit: {last['timestamp']}")
            print(f"Symptoms: {last['symptoms']}")
            print(f"Diagnosis: {', '.join(last['diagnoses'])}")
        else:
            print("Last Visit: None")
        print("-"*30)

        # Prepare History Context (Static)
        history_context = format_history_for_model(summary['full_history'])

        # Setup simple PromptTemplate using strict Llama-3-Instruct formatting
        prompt_template = PromptTemplate.from_template(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a medical assistant. You must analyze the CURRENT SYMPTOMS using the MEDICAL REFERENCE.\n"
            "Do NOT diagnose the patient with their PAST DIAGNOSES. The past diagnoses are only for context.\n"
            "Always end your response with \"Source: [URL]\".\n\n"
            "MEDICAL REFERENCE:\n{medical_context}\n\n"
            "PAST DIAGNOSES (Context only):\n{patient_history}\n\n"
            "CONVERSATION HISTORY:\n{chat_history_text}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "CURRENT SYMPTOMS: {input}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        # Setup HyDE Query Expansion Prompt
        hyde_prompt = PromptTemplate.from_template(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a medical diagnostic expert. Based on the provided symptoms, list 3 possible medical conditions or diseases. "
            "Return ONLY the names of the conditions, separated by commas. Do not include any other text.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Symptoms: {input}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        # Modern chain setup: pipe operator
        chain = prompt_template | llm | StrOutputParser()
        hyde_chain = hyde_prompt | llm | StrOutputParser()
        
        # Manually manage chat history
        chat_history = []

        print(f"\nChatting with {info['name']}'s assistant. (Type 'exit' to go back)")
        
        while True:
            user_input = input(f"\n[{info['name']}] > ").strip()
            if user_input.lower() == 'exit':
                break
            
            if not user_input:
                continue

            # Step 1: Query Expansion (HyDE)
            print("Analyzing symptoms for search expansion...", end="\r")
            try:
                expanded_query = hyde_chain.invoke({"input": user_input})
                # Combine original symptoms and the expanded conditions for a richer vector search
                search_query = f"{user_input} {expanded_query}"
            except Exception as e:
                search_query = user_input # Fallback if expansion fails

            # Step 2: RAG Retrieval
            print("Searching medical knowledge with expanded query...", end="\r")
            medical_context = retriever.retrieve(search_query)

            # Step 3: Final Generate Response
            print("Generating final response...                    ", end="\r")
            try:
                # Format chat history as text for the prompt
                chat_history_text = format_chat_history(chat_history)
                
                # Use invoke with the chain
                response = chain.invoke({
                    "input": user_input,
                    "patient_history": history_context,
                    "medical_context": medical_context,
                    "chat_history_text": chat_history_text
                })
                
                # Programmatically append the mandatory disclaimer to guarantee compliance
                disclaimer = "\n\n*** MANDATORY DISCLAIMER: Clinical validation by a physician is required before any medical action. ***"
                final_response = response.strip() + disclaimer
                
                print(f"\nAssistant: {final_response}")
                
                # Append to history
                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=final_response))

                # Log the interaction for auditing
                audit_logger.log_interaction(
                    patient_id=patient_id,
                    user_input=user_input,
                    context_used=medical_context,
                    history_used=history_context,
                    model_response=final_response
                )
                
            except Exception as e:
                print(f"\nError: {e}")

if __name__ == "__main__":
    main()

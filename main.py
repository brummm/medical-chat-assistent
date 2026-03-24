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
            formatted += f"Question: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            # Clean the disclaimer from history so it doesn't loop or clutter context
            clean_content = msg.content.replace("\n\n*** MANDATORY DISCLAIMER: Clinical validation by a physician is required before any medical action. ***", "")
            formatted += f"Answer: {clean_content}\n\n"
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

        # Setup simple PromptTemplate matching fine-tuning format exactly
        prompt_template = PromptTemplate.from_template(
            "Relevant medical information:\n{medical_context}\n\n"
            "Based on the information above, please answer the following question. "
            "Include the source URL in your answer.\n\n"
            "Patient History (For context only):\n{patient_history}\n\n"
            "Conversation History:\n{chat_history_text}\n\n"
            "Question: {input}\n"
            "Answer:"
        )
        
        # Modern chain setup: pipe operator
        chain = prompt_template | llm | StrOutputParser()
        
        # Manually manage chat history
        chat_history = []

        print(f"\nChatting with {info['name']}'s assistant. (Type 'exit' to go back)")
        
        while True:
            user_input = input(f"\n[{info['name']}] > ").strip()
            if user_input.lower() == 'exit':
                break
            
            if not user_input:
                continue

            # RAG Retrieval
            print("Searching medical knowledge...", end="\r")
            medical_context = retriever.retrieve(user_input)

            # Generate Response
            print("Generating response...        ", end="\r")
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

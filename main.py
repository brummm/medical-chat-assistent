import os
import sys
from datetime import datetime
from typing import Any, List, Optional

# LangChain imports
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

        # Setup modern LangChain Chain (LCEL)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "You are a professional medical assistant. Analyze the symptoms using the provided evidence.\n"
                "Format your response strictly as follows:\n"
                "Answer: [Detailed analysis]\n"
                "Source: [URL from the medical info]\n\n"
                "### SAFETY RULES:\n"
                "- NEVER prescribe directly.\n"
                "- MANDATORY DISCLAIMER: Your response MUST end with: 'Clinical validation by a physician is required before any medical action.' and the source URL.\n\n"
                "### PATIENT HISTORY:\n{patient_history}\n\n"
                "### MEDICAL REFERENCE INFO:\n{medical_context}<|eot_id|>"
            )),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "<|start_header_id|>user<|end_header_id|>\n\nQuestion: {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAnswer:")
        ])
        
        # Modern chain setup: pipe operator
        chain = prompt_template | llm | StrOutputParser()
        
        # Manually manage chat history for each patient session
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
                # Use invoke with the chain
                response = chain.invoke({
                    "input": user_input,
                    "patient_history": history_context,
                    "medical_context": medical_context,
                    "chat_history": chat_history
                })
                print(f"\nAssistant: {response}")
                
                # Append to history
                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=response))

                # Log the interaction for auditing
                audit_logger.log_interaction(
                    patient_id=patient_id,
                    user_input=user_input,
                    context_used=medical_context,
                    history_used=history_context,
                    model_response=response
                )
                
            except Exception as e:
                print(f"\nError: {e}")

if __name__ == "__main__":
    main()

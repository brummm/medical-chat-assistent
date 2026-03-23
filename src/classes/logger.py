import json
import logging
import os
from datetime import datetime

class MedicalAuditLogger:
    def __init__(self, log_dir="./persisted/logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Standard python logger for general events
        self.logger = logging.getLogger("medical_audit")
        self.logger.setLevel(logging.INFO)
        
        log_file = os.path.join(self.log_dir, "audit_trail.log")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_interaction(self, patient_id, user_input, context_used, history_used, model_response):
        """Records a full clinical interaction for audit purposes."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "patient_id": patient_id,
            "user_input": user_input,
            "context_used": context_used,
            "history_used": history_used,
            "model_response": model_response
        }
        
        # Log to a structured JSONL file for easy parsing/auditing
        jsonl_file = os.path.join(self.log_dir, "clinical_interactions.jsonl")
        with open(jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            
        self.logger.info(f"Interaction logged for Patient ID: {patient_id}")

import os
import sys
from datetime import date, timedelta
import random

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classes.database import MedicalDatabase

def populate_patients():
    db = MedicalDatabase()
    
    # Realistic patient data
    patients = [
        {"name": "Maria Silva", "sex": "Female", "dob": "1958-03-12"}, # 68 years
        {"name": "João Pereira", "sex": "Male", "dob": "1992-11-25"},  # 33 years
        {"name": "Ana Costa", "sex": "Female", "dob": "2015-06-08"},   # 10 years
        {"name": "Ricardo Santos", "sex": "Male", "dob": "1980-09-14"},# 45 years
        {"name": "Sofia Oliveira", "sex": "Female", "dob": "2002-01-30"} # 24 years
    ]
    
    print("Populating patients...")
    for p in patients:
        patient_id = db.add_patient(p["name"], p["sex"], p["dob"])
        print(f"Added Patient: {p['name']} (ID: {patient_id})")

if __name__ == "__main__":
    populate_patients()

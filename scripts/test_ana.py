import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.classes.model import MedicalModel

model = MedicalModel()
model.load_model()

prompt = """Relevant medical information:
--- MEDICAL REFERENCE 1 ---
Question: What are the symptoms of Crimean-Congo Hemorrhagic Fever (CCHF) ?
Answer: The onset of CCHF is sudden, with initial signs and symptoms including headache, high fever, back pain, joint pain, stomach pain, and vomiting. Red eyes, a flushed face, a red throat, and petechiae (red spots) on the palate are common.
Source: http://www.cdc.gov/vhf/crimean-congo/

Based on the information above, please answer the following question. Include the source URL in your answer.

Patient History (For context only):
- Date: 2025-03-01 03:36:04
  Symptoms: Periodic fever, joint pain, and skin rash recurring every few weeks.
  Diagnoses: TRAPS Syndrome
  Prescriptions: Etanercept injections, Prednisone for acute flares

Conversation History:

Question: The pacient is complaining about fever, fatigue, headache, nausea and vomiting.
Answer:"""

print("Running prompt...")
print(model.ask(prompt))

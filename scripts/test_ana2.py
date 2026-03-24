import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.classes.model import MedicalModel

model = MedicalModel()
model.load_model()

prompt = """Information:
Question: What are the symptoms of Crimean-Congo Hemorrhagic Fever (CCHF) ?
Answer: The onset of CCHF is sudden, with initial signs and symptoms including headache, high fever, back pain, joint pain, stomach pain, and vomiting. Red eyes, a flushed face, a red throat, and petechiae (red spots) on the palate are common.
Source: http://www.cdc.gov/vhf/crimean-congo/

Patient History:
- Date: 2025-03-01
  Symptoms: Periodic fever, joint pain, and skin rash.
  Diagnoses: TRAPS Syndrome

Conversation:
Question: The pacient feels nausea, headache and an itchy back. What can it be?
Answer: It could be anaphylaxis.

Question: Based on the Information and Patient History, the pacient is complaining about fever, fatigue, headache, nausea and vomiting. What can it be?
Answer:"""

print("Running prompt...")
print(model.ask(prompt))

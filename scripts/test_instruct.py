import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.classes.model import MedicalModel

model = MedicalModel()
model.load_model()

prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical assistant. You must analyze the CURRENT SYMPTOMS using the MEDICAL REFERENCE.
Do NOT diagnose the patient with their PAST DIAGNOSES. The past diagnoses are only for context.
Always end your response with "Source: [URL]".

MEDICAL REFERENCE:
Question: What are the symptoms of Crimean-Congo Hemorrhagic Fever (CCHF) ?
Answer: The onset of CCHF is sudden, with initial signs and symptoms including headache, high fever, back pain, joint pain, stomach pain, and vomiting. Red eyes, a flushed face, a red throat, and petechiae (red spots) on the palate are common.
Source: http://www.cdc.gov/vhf/crimean-congo/

PAST DIAGNOSES:
- TRAPS Syndrome<|eot_id|><|start_header_id|>user<|end_header_id|>

CURRENT SYMPTOMS: The pacient feels nausea, headache and an itchy back. What can it be?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

print("Running prompt...")
print(model.ask(prompt))

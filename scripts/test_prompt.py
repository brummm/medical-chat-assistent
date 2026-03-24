import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.classes.model import MedicalModel

model = MedicalModel()
model.load_model()

prompt = """Relevant medical information:
--- MEDICAL REFERENCE 1 ---
Question: How to diagnose Indigestion ?
Answer: To diagnose indigestion, the doctor asks about the person's current symptoms and medical history and performs a physical examination. The doctor may order x rays of the stomach or small intestine or use a tube with a light and camera on the end (an endoscope) to look inside the stomach.
Source: https://www.niddk.nih.gov/health-information/digestive-diseases/indigestion

Based on the information above, please answer the following question. Include the source URL in your answer.

Question: The patient has a stomach ache and blurry eyes
Answer:"""

print("Running prompt...")
print(model.ask(prompt))

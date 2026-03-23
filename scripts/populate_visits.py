import os
import sys
import random
from datetime import datetime, timedelta

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classes.database import MedicalDatabase

def populate_visits():
    db = MedicalDatabase()
    
    # Get all patients
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, sex, date_of_birth FROM patients")
        patients = [dict(row) for row in cursor.fetchall()]

    # Expanded pool of 30 unique medical scenarios for variety
    all_scenarios = [
        {"symptoms": "Red, itchy bumps on the arms and torso after eating shellfish.", "diagnoses": ["Hives (Urticaria)"], "prescriptions": ["Antihistamine (Cetirizine 10mg)", "Calamine lotion"]},
        {"symptoms": "Persistent dry mouth, difficulty swallowing dry foods, and frequent thirst.", "diagnoses": ["Xerostomia (Dry Mouth)"], "prescriptions": ["Artificial saliva substitute", "Pilocarpine 5mg"]},
        {"symptoms": "Lump found in left breast during self-examination, no pain.", "diagnoses": ["Suspicious breast mass"], "prescriptions": ["Referral for Mammogram", "Referral for Ultrasound"]},
        {"symptoms": "Difficulty seeing in low light conditions, bumping into objects at night.", "diagnoses": ["Congenital stationary night blindness"], "prescriptions": ["Vitamin A supplementation", "Referral to ophthalmology"]},
        {"symptoms": "Hip pain and limping, especially after walking long distances.", "diagnoses": ["Avascular necrosis of the femoral head"], "prescriptions": ["NSAIDs (Ibuprofen 400mg)", "Physical therapy referral"]},
        {"symptoms": "Hair appearing brittle and breaking easily, beaded appearance under magnification.", "diagnoses": ["Monilethrix"], "prescriptions": ["Minoxidil 2% topical solution", "Biotin supplements"]},
        {"symptoms": "Periodic fever, joint pain, and skin rash recurring every few weeks.", "diagnoses": ["TRAPS Syndrome"], "prescriptions": ["Etanercept injections", "Prednisone for acute flares"]},
        {"symptoms": "Severe muscle weakness and poor feeding in infant.", "diagnoses": ["Deoxyguanosine kinase deficiency"], "prescriptions": ["Liver support therapy"]},
        {"symptoms": "Difficulty with speech and communication, stuttering noticed.", "diagnoses": ["Speech and Communication Disorder"], "prescriptions": ["Speech therapy referral"]},
        {"symptoms": "Abnormal vaginal bleeding and pelvic pressure.", "diagnoses": ["Uterine Sarcoma suspicion"], "prescriptions": ["Biopsy referral"]},
        {"symptoms": "Chronic diarrhea with blood and abdominal discomfort.", "diagnoses": ["Ulcerative Colitis"], "prescriptions": ["Mesalamine", "Dietary adjustments"]},
        {"symptoms": "Shortness of breath during routine activity and tiredness.", "diagnoses": ["Pulmonary Hypertension"], "prescriptions": ["Sildenafil", "Oxygen therapy"]},
        {"symptoms": "Scaly, itchy patches on elbows and knees.", "diagnoses": ["Psoriasis"], "prescriptions": ["Topical corticosteroids", "Vitamin D analogues"]},
        {"symptoms": "Severe reaction to anesthesia with muscle rigidity and fever.", "diagnoses": ["Malignant Hyperthermia (CCD related)"], "prescriptions": ["Dantrolene", "Avoidance of trigger agents"]},
        {"symptoms": "Clouding of the lens and abnormally small eyeballs.", "diagnoses": ["Microphthalmia with Cataract"], "prescriptions": ["Surgical consultation", "Visual aids"]},
        {"symptoms": "Rapidly progressive dementia and muscle jerks.", "diagnoses": ["Creutzfeldt-Jakob disease suspicion"], "prescriptions": ["Palliative care referral", "Neurology consultation"]},
        {"symptoms": "Vomiting and lethargy after mosquito bite in summer.", "diagnoses": ["La Crosse encephalitis"], "prescriptions": ["Supportive care", "Fluid management"]},
        {"symptoms": "Recurrent seizures and loss of movement abilities in child.", "diagnoses": ["Alpers-Huttenlocher syndrome"], "prescriptions": ["Anticonvulsants", "Liver function monitoring"]},
        {"symptoms": "Loss of bone tissue in hands and feet, joint pain.", "diagnoses": ["Multicentric osteolysis (MONA)"], "prescriptions": ["Bisphosphonates", "Pain management"]},
        {"symptoms": "Excessive sweating and heat intolerance.", "diagnoses": ["Hyperhidrosis"], "prescriptions": ["Aluminum chloride topical", "Glycopyrrolate"]},
        {"symptoms": "Chronic fatigue, joint pain, and butterfly rash on face.", "diagnoses": ["Systemic Lupus Erythematosus"], "prescriptions": ["Hydroxychloroquine", "Sun protection"]},
        {"symptoms": "Tremors in hands and difficulty with fine motor tasks.", "diagnoses": ["Essential Tremor"], "prescriptions": ["Propranolol", "Occupational therapy"]},
        {"symptoms": "Severe headache with light sensitivity and nausea.", "diagnoses": ["Migraine with Aura"], "prescriptions": ["Sumatriptan", "Magnesium supplements"]},
        {"symptoms": "Frequent urination and excessive thirst.", "diagnoses": ["Diabetes Mellitus Type 2"], "prescriptions": ["Metformin", "Glucose monitoring"]},
        {"symptoms": "Persistent cough and wheezing, worse at night.", "diagnoses": ["Asthma"], "prescriptions": ["Albuterol inhaler", "Fluticasone"]},
        {"symptoms": "Chest pain radiating to the left arm during exertion.", "diagnoses": ["Stable Angina"], "prescriptions": ["Nitroglycerin", "Aspirin 81mg"]},
        {"symptoms": "Yellowing of the eyes and dark urine.", "diagnoses": ["Hepatitis A"], "prescriptions": ["Rest and hydration", "Avoidance of hepatotoxic drugs"]},
        {"symptoms": "Painful swelling of the big toe joint.", "diagnoses": ["Gout"], "prescriptions": ["Allopurinol", "Colchicine"]},
        {"symptoms": "Difficulty falling asleep and staying asleep.", "diagnoses": ["Chronic Insomnia"], "prescriptions": ["Melatonin", "Sleep hygiene counseling"]},
        {"symptoms": "Sudden loss of vision in one eye, painless.", "diagnoses": ["Retinal Artery Occlusion"], "prescriptions": ["Emergency ophthalmology referral", "Blood thinners"]}
    ]

    print("Populating unique visits for each patient...")
    for patient in patients:
        # Determine number of visits (1 to 10)
        num_visits = random.randint(1, 10)
        
        # Pick N unique scenarios for THIS patient to avoid repetition
        patient_scenarios = random.sample(all_scenarios, num_visits)
        
        print(f"Adding {num_visits} unique visits for {patient['name']}...")
        
        # Generate N random timestamps over the last 2 years and sort them
        current_date = datetime.now()
        timestamps = []
        for _ in range(num_visits):
            days_ago = random.randint(0, 730)
            seconds_ago = random.randint(0, 86400)
            visit_time = current_date - timedelta(days=days_ago, seconds=seconds_ago)
            timestamps.append(visit_time)
        
        timestamps.sort() # Chronological order

        for i, visit_time in enumerate(timestamps):
            scenario = patient_scenarios[i]
            formatted_time = visit_time.strftime("%Y-%m-%d %H:%M:%S")
            
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO visits (patient_id, timestamp, symptoms) VALUES (?, ?, ?)",
                    (patient['id'], formatted_time, scenario['symptoms'])
                )
                visit_id = cursor.lastrowid
                
                for diag in scenario['diagnoses']:
                    cursor.execute(
                        "INSERT INTO diagnoses (visit_id, description) VALUES (?, ?)",
                        (visit_id, diag)
                    )
                
                for pres in scenario['prescriptions']:
                    cursor.execute(
                        "INSERT INTO prescriptions (visit_id, description) VALUES (?, ?)",
                        (visit_id, pres)
                    )
                conn.commit()

    print("Visits population complete.")

if __name__ == "__main__":
    populate_visits()

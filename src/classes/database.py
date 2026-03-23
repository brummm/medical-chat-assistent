import sqlite3
import os

class MedicalDatabase:
    def __init__(self, db_path="./persisted/medical_records.db"):
        self.db_path = db_path
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes the database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Patients table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    sex TEXT,
                    date_of_birth DATE
                )
            ''')
            
            # Visits table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symptoms TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients (id)
                )
            ''')
            
            # Diagnoses table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS diagnoses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visit_id INTEGER NOT NULL,
                    description TEXT NOT NULL,
                    FOREIGN KEY (visit_id) REFERENCES visits (id)
                )
            ''')
            
            # Prescriptions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prescriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visit_id INTEGER NOT NULL,
                    description TEXT NOT NULL,
                    FOREIGN KEY (visit_id) REFERENCES visits (id)
                )
            ''')
            
            conn.commit()
        print(f"Database initialized at {self.db_path}")

    def add_patient(self, name, sex, dob):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO patients (name, sex, date_of_birth) VALUES (?, ?, ?)",
                (name, sex, dob)
            )
            return cursor.lastrowid

    def add_visit(self, patient_id, symptoms, diagnoses=None, prescriptions=None):
        """Adds a visit along with its diagnoses and prescriptions."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO visits (patient_id, symptoms) VALUES (?, ?)",
                (patient_id, symptoms)
            )
            visit_id = cursor.lastrowid
            
            if diagnoses:
                for diag in diagnoses:
                    cursor.execute(
                        "INSERT INTO diagnoses (visit_id, description) VALUES (?, ?)",
                        (visit_id, diag)
                    )
            
            if prescriptions:
                for pres in prescriptions:
                    cursor.execute(
                        "INSERT INTO prescriptions (visit_id, description) VALUES (?, ?)",
                        (visit_id, pres)
                    )
            
            conn.commit()
            return visit_id

    def get_patient_history(self, patient_id):
        """Retrieves full history for a patient."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get visits
            cursor.execute("SELECT * FROM visits WHERE patient_id = ? ORDER BY timestamp DESC", (patient_id,))
            visits = [dict(row) for row in cursor.fetchall()]
            
            for visit in visits:
                # Get diagnoses
                cursor.execute("SELECT description FROM diagnoses WHERE visit_id = ?", (visit['id'],))
                visit['diagnoses'] = [row['description'] for row in cursor.fetchall()]
                
                # Get prescriptions
                cursor.execute("SELECT description FROM prescriptions WHERE visit_id = ?", (visit['id'],))
                visit['prescriptions'] = [row['description'] for row in cursor.fetchall()]
                
        return visits

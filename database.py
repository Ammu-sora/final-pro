import sqlite3
import pandas as pd
import numpy as np

def create_connection():
    return sqlite3.connect("health_predictions.db", check_same_thread=False)

def init_db():
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()

    # Create diabetes_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diabetes_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Pregnancies INTEGER,
            Glucose INTEGER,
            BloodPressure INTEGER,
            SkinThickness INTEGER,
            Insulin INTEGER,
            BMI REAL,
            DiabetesPedigreeFunction REAL,
            Age INTEGER,
            Prediction TEXT,
            Precaution TEXT
        )
    ''')

    # Create heart_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS heart_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            sex INTEGER,
            cp INTEGER,
            trestbps INTEGER,
            chol INTEGER,
            fbs INTEGER,
            restecg INTEGER,
            thalach INTEGER,
            exang INTEGER,
            oldpeak REAL,
            slope INTEGER,
            ca INTEGER,
            thal INTEGER,
            prediction TEXT,
            precaution TEXT
        )
    ''')

    # Create parkinsons_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parkinsons_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fo REAL,
            fhi REAL,
            flo REAL,
            jitter_percent REAL,
            jitter_abs REAL,
            rap REAL,
            ppq REAL,
            ddp REAL,
            shimmer REAL,
            shimmer_db REAL,
            apq3 REAL,
            apq5 REAL,
            apq REAL,
            dda REAL,
            nhr REAL,
            hnr REAL,
            rpde REAL,
            dfa REAL,
            spread1 REAL,
            spread2 REAL,
            d2 REAL,
            PPE REAL,
            prediction TEXT,
            precaution TEXT
        )
    ''')

    # Create calories_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS calories_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Gender TEXT,
            Age INTEGER,
            Height REAL,
            Weight REAL,
            Duration INTEGER,
            Heart_Rate INTEGER,
            Body_Temp REAL,
            Calories_Burnt REAL
        )
    ''')

    conn.commit()
    conn.close()

def create_tables():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS diabetes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pregnancies REAL, glucose REAL, blood_pressure REAL, skin_thickness REAL,
        insulin REAL, bmi REAL, dpf REAL, age REAL, prediction TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS heart_disease (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        age REAL, sex REAL, chest_pain REAL, blood_pressure REAL,
        cholestoral REAL, blood_sugar REAL, ecg REAL, heart_rate REAL,
        angina REAL, st_depression REAL, st_segment REAL, flourosopy REAL,
        defect REAL, prediction TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS parkinsons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fo REAL, fhi REAL, flo REAL, jitter_percent REAL, jitter_abs REAL,
        rap REAL, ppq REAL, ddp REAL, shimmer REAL, shimmer_db REAL,
        shimmer_apq3 REAL, shimmer_apq5 REAL, mdvp_apq REAL, shimmer_dda REAL,
        nhr REAL, hnr REAL, rpde REAL, dfa REAL, spread1 REAL, spread2 REAL,
        d2 REAL, ppe REAL, prediction TEXT
    )
    ''')
    cursor.execute("DROP TABLE IF EXISTS parkinsons_data")


    cursor.execute('''
    CREATE TABLE IF NOT EXISTS calories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gender REAL, age REAL, height REAL, weight REAL,
        duration REAL, heart_rate REAL, body_temp REAL, calories_burnt REAL
    )
    ''')

    conn.commit()
    conn.close()

def insert_diabetes(data, prediction):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()

    # Destructure data list
    (Pregnancies, Glucose, BloodPressure, SkinThickness,
     Insulin, BMI, DiabetesPedigreeFunction, Age) = data

    precaution = (
        "Maintain a balanced diet, exercise regularly, monitor glucose, "
        "take medication on time, stay hydrated, and sleep well." if prediction == 'Diabetic'
        else "Continue a healthy lifestyle, regular checkups, stay active and eat well."
    )

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diabetes_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Pregnancies REAL,
            Glucose REAL,
            BloodPressure REAL,
            SkinThickness REAL,
            Insulin REAL,
            BMI REAL,
            DiabetesPedigreeFunction REAL,
            Age REAL,
            Prediction TEXT,
            Precaution TEXT
        )
    ''')

    cursor.execute('''
        INSERT INTO diabetes_data (
            Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
            BMI, DiabetesPedigreeFunction, Age, Prediction, Precaution
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age, prediction, precaution
    ))

    conn.commit()
    conn.close()


def insert_heart(data, prediction):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()

    precaution = (
        "Reduce salt and fat intake, exercise regularly, manage stress, avoid smoking, monitor cholesterol and BP."
        if prediction == 'At Risk' else
        "Maintain a healthy lifestyle, regular checkups, balanced diet, and exercise."
    )

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS heart_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age REAL, sex REAL, cp REAL, trestbps REAL, chol REAL,
            fbs REAL, restecg REAL, thalach REAL, exang REAL, oldpeak REAL,
            slope REAL, ca REAL, thal REAL,
            Prediction TEXT, Precaution TEXT
        )
    ''')

    # Flatten data
    if isinstance(data, np.ndarray):
        data = data.flatten().tolist()
    elif isinstance(data[0], (list, tuple, np.ndarray)):
        data = list(data[0])

    if len(data) != 13:
        print("⚠️ insert_heart(): Invalid data length:", len(data))
        conn.close()
        return

    try:
        cursor.execute('''
            INSERT INTO heart_data (
                age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal, Prediction, Precaution
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (*data, prediction, precaution))
        conn.commit()
        print("✅ Inserted:", (*data, prediction, precaution))
        # Show last row
        cursor.execute("SELECT * FROM heart_data ORDER BY id DESC LIMIT 1")
        print("Last inserted row:", cursor.fetchone())
    except Exception as e:
        print("❌ Error inserting heart data:", e)
    finally:
        conn.close()


import sqlite3
import numpy as np

def insert_parkinsons(data, prediction):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()

    precaution = (
        "Follow a healthy diet, stay active, take prescribed medications, consult a neurologist regularly."
        if prediction == "Parkinson's Disease" else
        "Maintain regular checkups, stay mentally and physically active, and follow a balanced diet."
    )

    # ✅ Create the table only once (if not already exists)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parkinsons_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            MDVP_Fo REAL, MDVP_Fhi REAL, MDVP_Flo REAL, MDVP_Jitter_percent REAL,
            MDVP_Jitter_Abs REAL, MDVP_RAP REAL, MDVP_PPQ REAL, Jitter_DDP REAL,
            MDVP_Shimmer REAL, MDVP_Shimmer_dB REAL, Shimmer_APQ3 REAL, Shimmer_APQ5 REAL,
            MDVP_APQ REAL, Shimmer_DDA REAL, NHR REAL, HNR REAL,
            RPDE REAL, DFA REAL, spread1 REAL, spread2 REAL,
            D2 REAL, PPE REAL,
            Prediction TEXT, Precaution TEXT
        )
    ''')

    # 🧾 Flatten input if needed
    if isinstance(data, np.ndarray):
        data = data.flatten().tolist()
    elif isinstance(data[0], (list, tuple, np.ndarray)):
        data = list(data[0])

    if len(data) != 22:
        print("❌ insert_parkinsons(): Expected 22 values, got", len(data))
        conn.close()
        return

    # 📥 Insert
    cursor.execute('''
        INSERT INTO parkinsons_data (
            MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter_percent,
            MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP,
            MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5,
            MDVP_APQ, Shimmer_DDA, NHR, HNR,
            RPDE, DFA, spread1, spread2,
            D2, PPE, Prediction, Precaution
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (*data, prediction, precaution))

    conn.commit()
    conn.close()


def insert_calories(data, prediction):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()

    precaution = (
        "Maintain a balanced diet, monitor your calorie intake, stay hydrated, and exercise regularly."
        if prediction == "High Calories Burnt" else
        "Consider increasing your physical activity and eating healthy to improve calorie burn."
    )

    # ✅ Create the table if it does not exist (don’t drop!)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS calories_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Gender TEXT, Age REAL, Height REAL, Weight REAL,
            Duration REAL, Heart_Rate REAL, Body_Temp REAL,
            Precaution TEXT
        )
    ''')

    # 🧾 Flatten input if it's wrapped in a list or NumPy array
    if isinstance(data, np.ndarray):
        data = data.flatten().tolist()
    elif isinstance(data[0], (list, tuple, np.ndarray)):
        data = list(data[0])

    if len(data) != 7:
        print("❌ insert_calories(): Expected 7 input values, got", len(data))
        conn.close()
        return

    # 📥 Insert into database
    cursor.execute('''
        INSERT INTO calories_data (
            Gender, Age, Height, Weight,
            Duration, Heart_Rate, Body_Temp,
             Precaution
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (*data, precaution))

    conn.commit()
    conn.close()


    
def fetch_all_diabetes():
    conn = sqlite3.connect('user_data.db')
    df = pd.read_sql_query("SELECT * FROM diabetes_data", conn)
    conn.close()
    return df

def fetch_all_heart():
    conn = sqlite3.connect('user_data.db')
    df = pd.read_sql_query("SELECT * FROM heart_data", conn)
    conn.close()
    return df

def fetch_all_parkinsons():
    conn = sqlite3.connect('user_data.db')
    df = pd.read_sql_query("SELECT * FROM parkinsons_data", conn)
    conn.close()
    return df

def fetch_all_calories():
    conn = sqlite3.connect('user_data.db')
    df = pd.read_sql_query("SELECT * FROM calories_data", conn)
    conn.close()
    return df








    
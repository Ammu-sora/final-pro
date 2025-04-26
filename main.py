import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import streamlit.components.v1 as components
import time
from database import init_db
init_db()
from database import create_tables
create_tables()
import pandas as pd
from database import create_connection
from database import fetch_all_diabetes, fetch_all_heart, fetch_all_parkinsons, fetch_all_calories
import plotly.express as px


# Set page config FIRST (must be before any Streamlit commands)
st.set_page_config(page_title="Health Predictor", page_icon="🧠", layout="wide")

# Now define your home page UI
def home_page():
        st.image("D:/Multiple Disease Detection Model/assets/3.png", use_container_width=True)

    
def dashboard_page():
    st.markdown("""
    <div style='background-color:#e0f7fa; padding:20px; border-radius:10px'>
        <h2 style='color:#00796b;'>📊 Health Prediction Dashboard</h2>
        <p style='font-size:16px;'>Review all stored data and predictions from previous inputs.</p>
    </div>
    """, unsafe_allow_html=True)

    conn = create_connection()

    # Diabetes Table
    st.subheader("🩸 Diabetes Predictions")
    diabetes_df = pd.read_sql_query("SELECT * FROM diabetes", conn)
    st.dataframe(diabetes_df)

    # Heart Disease Table
    st.subheader("❤️ Heart Disease Predictions")
    heart_df = pd.read_sql_query("SELECT * FROM heart_disease", conn)
    st.dataframe(heart_df)

    # Parkinson’s Table
    st.subheader("🧠 Parkinson's Predictions")
    parkinsons_df = pd.read_sql_query("SELECT * FROM parkinsons", conn)
    st.dataframe(parkinsons_df)

    # Calories Table
    st.subheader("🔥 Calories Burnt Predictions")
    calories_df = pd.read_sql_query("SELECT * FROM calories", conn)
    st.dataframe(calories_df)

    conn.close()

# Optional app-wide background color
st.markdown("""
<style>
/* Page gradient background */
body {
    background: linear-gradient(to right, #eef2f3, #8e9eab);
    animation: fadeInBody 1s ease-in;
}

/* Form slide-in animation */
section[data-testid="stForm"] {
    animation: slideIn 0.7s ease forwards;
}

/* Animated glowing input borders */
.stTextInput>div>input {
    border-radius: 10px;
    padding: 10px;
    border: 2px solid #ccc;
    transition: all 0.3s ease;
}
.stTextInput>div>input:focus {
    border-color: #6c63ff;
    box-shadow: 0 0 10px #6c63ff;
}

/* Button animation */
.stButton>button {
    border-radius: 10px;
    background-color: #6c63ff;
    color: white;
    padding: 10px 20px;
    font-weight: bold;
    transition: transform 0.2s ease, background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #5a54d1;
    transform: scale(1.05);
}

/* Result pulse animation */
.result-box {
    animation: pulse 1.5s infinite;
}

/* Animations */
@keyframes fadeInBody {
    from { opacity: 0; }
    to { opacity: 1; }
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(108, 99, 255, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(108, 99, 255, 0); }
    100% { box-shadow: 0 0 0 0 rgba(108, 99, 255, 0); }
}
</style>
""", unsafe_allow_html=True)


# Load models
models = {
    'diabetes': pickle.load(open('D:/Multiple Disease Detection Model/saved models/diabetes_model.sav', 'rb')),
    'heart': pickle.load(open('D:/Multiple Disease Detection Model/saved models/heart_disease_model.sav', 'rb')),
    'parkinsons': pickle.load(open('D:/Multiple Disease Detection Model/saved models/parkinsons_model.sav', 'rb')),
    'calories': pickle.load(open('D:/Multiple Disease Detection Model/saved models/calories_burned.sav', 'rb')),
}

# Utility function
def safe_float(value):
    try:
        return float(value.strip()) if value.strip() else 0.0
    except ValueError:
        return 0.0

# Sidebar navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2904/2904978.png", width=100)
    
    selected = option_menu(
        'Multiple Disease & Fitness Prediction',
        ['Home','Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Calories Burnt Prediction','Dashboard','Data Visualiser'],
        icons=['house','activity','heart','person','person-gear','bar-chart','clipboard-data'],
        menu_icon="stethoscope",
        default_index=0,
        styles={
            "container": {"padding": "5px"},
            "icon": {"color": "orange", "font-size": "20px"},
            "nav-link": {"font-size": "18px", "text-align": "left", "margin":"2px"},
            "nav-link-selected": {"background-color": "#6c63ff"},
        }
        
    )

def dashboard_page():
    st.title("📊 Disease Prediction Dashboard")

    # Load Data
    diabetes_df = fetch_all_diabetes()
    heart_df = fetch_all_heart()
    parkinsons_df = fetch_all_parkinsons()
    calories_df = fetch_all_calories()

    # Convert timestamps
    for df in [diabetes_df, heart_df, parkinsons_df, calories_df]:
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

    # === Home Summary Cards ===
    st.markdown("## 🏠 Dashboard Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🩸 Diabetes", len(diabetes_df))
    col2.metric("❤️ Heart", len(heart_df))
    col3.metric("🧠 Parkinson's", len(parkinsons_df))
    col4.metric("🔥 Calories", len(calories_df))

    # === Navigate to Tabs ===
    st.markdown("## 📄 Detailed Reports")
    tabs = st.tabs(["🩸 Diabetes", "❤️ Heart", "🧠 Parkinson's", "🔥 Calories"])

    # Reusable function for each tab
    def show_disease_tab(df, disease_name):
        st.subheader(f"{disease_name} Predictions")

        # Filters (gender and date)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            min_date = df['timestamp'].min().date()
            max_date = df['timestamp'].max().date()
            date_range = st.date_input(f"📅 Date Range for {disease_name}", [min_date, max_date])
            df = df[(df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1])]

        if 'gender' in df.columns:
            gender_filter = st.selectbox(f"🚻 Gender Filter for {disease_name}", ["All", "Male", "Female"])
            if gender_filter != "All":
                df = df[df['gender'].str.lower() == gender_filter.lower()]

        # Line Chart over Time
        if 'timestamp' in df.columns:
            count_by_date = df['timestamp'].dt.date.value_counts().sort_index()
            fig = px.line(x=count_by_date.index, y=count_by_date.values,
                          labels={'x': 'Date', 'y': 'Predictions'},
                          title=f"{disease_name} Predictions Over Time")
            st.plotly_chart(fig, use_container_width=True)

        # Data Table and Download
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(f"📥 Download {disease_name} Data", csv, f"{disease_name.lower()}_data.csv", "text/csv")

    # Each tab
    with tabs[0]:
        show_disease_tab(diabetes_df, "🩸 Diabetes")

    with tabs[1]:
        show_disease_tab(heart_df, "❤️ Heart Disease")

    with tabs[2]:
        show_disease_tab(parkinsons_df, "🧠 Parkinson's")

    with tabs[3]:
        show_disease_tab(calories_df, "🔥 Calories Burnt")

# Disease Visualizer Page
def data_visualiser_page():
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st

    st.title("👩🏻‍💻 Data Visualizer")

    # Setting up dataset path
    working_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(working_dir, "dataset")

    # List CSV files
    files_list = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    selected_file = st.selectbox("📂 Select a CSV file to visualize", files_list, index=None)

    if selected_file:
        file_path = os.path.join(folder_path, selected_file)
        df = pd.read_csv(file_path)

        st.markdown("### 🧾 Preview of the Data")
        st.dataframe(df.head(), use_container_width=True)

        st.markdown("### 📊 Visualization Settings")

        columns = df.columns.tolist()
        col1, col2 = st.columns(2)

        with col1:
            x_axis = st.selectbox("Select the X-axis", options=columns + ['None'], index=0)

        with col2:
            y_axis = st.selectbox("Select the Y-axis", options=columns + ['None'], index=1)

        plot_list = ["Line Plot", "Bar Chart", "Scatter Plot", "Distribution Plot", "Count Plot"]
        selected_plot = st.selectbox("Select a Plot Type", options=plot_list, index=0)

        if st.button("📈 Generate Plot"):
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.set(style="whitegrid")

                if selected_plot == 'Line Plot':
                    sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax, color='mediumblue')

                elif selected_plot == 'Bar Chart':
                    sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax, palette='Set2')

                elif selected_plot == 'Scatter Plot':
                    sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax, color='darkorange', edgecolor='black')

                elif selected_plot == 'Distribution Plot':
                    sns.histplot(df[x_axis], kde=True, ax=ax, color='purple')

                elif selected_plot == 'Count Plot':
                    sns.countplot(data=df, x=x_axis, ax=ax, palette='Set3')

                ax.tick_params(axis='x', labelsize=10)
                ax.tick_params(axis='y', labelsize=10)
                ax.set_title(f"{selected_plot} of {y_axis} vs {x_axis}", fontsize=12)
                ax.set_xlabel(x_axis, fontsize=10)
                ax.set_ylabel(y_axis if y_axis != 'None' else "Count", fontsize=10)

                st.pyplot(fig)

            except Exception as e:
                st.error(f"⚠️ Error generating plot: {e}")

from database import insert_diabetes
def diabetes_prediction_page():
    st.markdown("""
    <div style='background-color:#f9f9f9; padding:20px; border-radius:10px'>
        <h2 style='color:#6c63ff;'>🩸 Diabetes Prediction</h2>
        <p style='font-size:16px;'>Fill in the required health parameters to assess the risk of <strong>Diabetes</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📝 Enter Patient Details", unsafe_allow_html=True)

    with st.form(key="diabetes_form"):
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                Pregnancies = st.text_input('🤰 Number of Pregnancies *')
                SkinThickness = st.text_input('📏 Skin Thickness (mm) *')
                DiabetesPedigreeFunction = st.text_input('🧬 Pedigree Function *')
            with col2:
                Glucose = st.text_input('🍭 Glucose Level (mg/dL) *')
                Insulin = st.text_input('💉 Insulin Level (IU/mL) *')
                Age = st.text_input('🎂 Age *')
            with col3:
                BloodPressure = st.text_input('💓 Blood Pressure (mm Hg) *')
                BMI = st.text_input('⚖️ BMI (kg/m²) *')

        submitted = st.form_submit_button('🔍 Predict Diabetes')

    if submitted:
        # Validate all fields are filled
        fields = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                  Insulin, BMI, DiabetesPedigreeFunction, Age]

        if any(field.strip() == "" for field in fields):
            st.error("🚨 Please fill in **all required fields** to proceed.")
        else:
            input_values = [
                        safe_float(Pregnancies), safe_float(Glucose), safe_float(BloodPressure),
                        safe_float(SkinThickness), safe_float(Insulin), safe_float(BMI),
                        safe_float(DiabetesPedigreeFunction), safe_float(Age)
                           ]

            # Predict diabetes
            diab_prediction = models['diabetes'].predict([input_values])
            prediction_result = 'Diabetic' if diab_prediction[0] == 0 else 'Not Diabetic'
            # ✅ Store in database
            insert_diabetes(input_values, prediction_result)

            st.divider()
            if diab_prediction[0] == 0:
                st.error('⚠️ The person is **Diabetic**')
                st.info("""
                ### 🩺 Precautions/Suggestions:
                - Maintain a balanced diet low in sugar and carbs.
                - Exercise regularly (30 minutes daily).
                - Monitor blood glucose levels.
                - Take prescribed medication on time.
                - Stay hydrated and get enough sleep.
                """)
            else:
                st.success('✅ The person is **not Diabetic**')
                st.info("""
                ### 🧘 General Health Tips:
                - Continue a healthy lifestyle.
                - Regular health checkups.
                - Stay active and eat nutritious food.
                """)
                st.balloons()
                       

from database import insert_heart 
def heart_disease_prediction_page():
    st.markdown("""
    <div style='background-color:#f9f9f9; padding:20px; border-radius:10px'>
        <h2 style='color:#e63946;'>❤️ Heart Disease Prediction</h2>
        <p style='font-size:16px;'>Fill in the health parameters to assess the risk of <strong>Heart Disease</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📝 Enter Patient Details", unsafe_allow_html=True)

    with st.form(key="heart_form"):
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                Age = st.text_input('🎂 Age *')
                BloodPressure = st.text_input('💓 Resting Blood Pressure *')
                Electrocardiographics = st.text_input('🧠 Resting ECG Results *')
                STdepressioninduced = st.text_input('📉 ST Depression by Exercise *')
                Flourosopy = st.text_input('🩻 Fluoroscopy (Major Vessels) *')
            with col2:
                Sex = st.text_input('⚥ Sex (1=Male, 0=Female) *')
                SerumCholestoral = st.text_input('🧪 Serum Cholestoral (mg/dL) *')
                HeartRate = st.text_input('💗 Max Heart Rate Achieved *')
                STsegment = st.text_input('📈 Slope of ST Segment *')
                ReversableDefect = st.text_input('🧬 Reversible Defect (0-2) *')
            with col3:
                ChestPain = st.text_input('💢 Chest Pain Type *')
                BloodSugar = st.text_input('🩸 Fasting Blood Sugar >120 mg/dL *')
                InducedAngina = st.text_input('🏃 Exercise Induced Angina *')

        submitted = st.form_submit_button('🔍 Predict Heart Disease')

    if submitted:
        # Validation
        fields = [Age, Sex, ChestPain, BloodPressure, SerumCholestoral, BloodSugar,
                  Electrocardiographics, HeartRate, InducedAngina, STdepressioninduced,
                  STsegment, Flourosopy, ReversableDefect]

        if any(field.strip() == "" for field in fields):
            st.error("🚨 Please fill in **all required fields** to proceed.")
        else:
            input_data = np.array([[safe_float(Age), safe_float(Sex), safe_float(ChestPain),
                                    safe_float(BloodPressure), safe_float(SerumCholestoral), safe_float(BloodSugar),
                                    safe_float(Electrocardiographics), safe_float(HeartRate), safe_float(InducedAngina),
                                    safe_float(STdepressioninduced), safe_float(STsegment), safe_float(Flourosopy),
                                    safe_float(ReversableDefect)]])
 
            heart_prediction = models['heart'].predict(input_data)
            prediction_result = 'Healthy' if heart_prediction[0] == 1 else 'At Risk'
            # ✅ Store data in the database
            insert_heart(input_data, prediction_result)
            st.divider()
            if heart_prediction[0] ==0:
                st.error('⚠️ The person has **Heart Disease**')
                st.info("""
                ### 🩺 Precautions/Suggestions:
                - Reduce salt and saturated fat intake.
                - Stop smoking and limit alcohol.
                - Take BP and cholesterol-lowering medications.
                - Exercise under medical supervision.
                - Manage stress and get checkups.
                """)
            else:
                st.success('✅ The person does **not** have Heart Disease')
                st.markdown("""
                <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
                <script>
                confetti({
                  particleCount: 150,
                  spread: 70,
                  origin: { y: 0.6 }
                });
                </script>
                """, unsafe_allow_html=True)
                st.info("""
                ### ❤️ General Health Tips:
                - Maintain a heart-healthy lifestyle.
                - Exercise regularly and eat clean.
                - Avoid stress and schedule regular checkups.
                """)
                st.balloons()

from database import insert_parkinsons 

def parkinsons_prediction_page():
    st.markdown("""
    <div style='background-color:#f9f9f9; padding:20px; border-radius:10px'>
        <h2 style='color:#6a4c93;'>🧠 Parkinson's Prediction</h2>
        <p style='font-size:16px;'>Enter voice frequency and jitter parameters to predict <strong>Parkinson's Disease</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form(key="parkinsons_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            fo = st.text_input('🎵 Fo (Fundamental Frequency - Hz) *')
            jitter_percent = st.text_input('📉 Jitter (%) *')
            rap = st.text_input('🔁 RAP (Relative Average Perturbation) *')
            shimmer = st.text_input('🌟 Shimmer *')
            shimmer_apq5 = st.text_input('🌟 APQ5 (Shimmer) *')
            shimmer_dda = st.text_input('🌟 DDA (Shimmer Derivative) *')
            hnr = st.text_input('🔊 HNR (Harmonic-to-Noise Ratio) *')
            spread1 = st.text_input('🌀 Spread1 *')

        with col2:
            fhi = st.text_input('🎶 Fhi (Highest Frequency - Hz) *')
            jitter_abs = st.text_input('📉 Jitter (Absolute) *')
            ppq = st.text_input('🔁 PPQ *')
            shimmer_db = st.text_input('🌟 Shimmer (dB) *')
            mdvp_apq = st.text_input('📊 APQ (Amplitude Perturbation Quotient) *')
            nhr = st.text_input('🔇 NHR (Noise-to-Harmonic Ratio) *')
            rpde = st.text_input('📈 RPDE (Recurrence Period Density Entropy) *')
            spread2 = st.text_input('🌀 Spread2 *')

        with col3:
            flo = st.text_input('🎵 Flo (Lowest Frequency - Hz) *')
            ddp = st.text_input('🔁 DDP *')
            shimmer_apq3 = st.text_input('🌟 APQ3 *')
            dfa = st.text_input('📐 DFA (Detrended Fluctuation Analysis) *')
            d2 = st.text_input('📊 D2 (Correlation Dimension) *')
            ppe = st.text_input('🧩 PPE (Pitch Period Entropy) *')

        submitted = st.form_submit_button("🔍 Predict Parkinson's")

    if submitted:
        fields = [fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                  shimmer, shimmer_db, shimmer_apq3, shimmer_apq5, mdvp_apq,
                  shimmer_dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]

        if any(field.strip() == "" for field in fields):
            st.error("🚨 Please fill in **all required fields** to proceed.")
        else:
            input_data = np.array([[safe_float(fo), safe_float(fhi), safe_float(flo), safe_float(jitter_percent),
                                    safe_float(jitter_abs), safe_float(rap), safe_float(ppq), safe_float(ddp),
                                    safe_float(shimmer), safe_float(shimmer_db), safe_float(shimmer_apq3),
                                    safe_float(shimmer_apq5), safe_float(mdvp_apq), safe_float(shimmer_dda),
                                    safe_float(nhr), safe_float(hnr), safe_float(rpde), safe_float(dfa),
                                    safe_float(spread1), safe_float(spread2), safe_float(d2), safe_float(ppe)]])

            # ✅ Model prediction
            prediction = models['parkinsons'].predict(input_data)
            prediction_result = 'Parkinson’s Detected' if prediction[0] == 1 else 'No Parkinson’s'

            # ✅ Store in database
            insert_parkinsons(input_data, prediction_result)

            st.divider()
            if prediction[0] == 1:
                st.error("⚠️ The person has **Parkinson’s Disease**")
                st.info("""
                ### 🩺 Precautions/Suggestions:
                - Consult a neurologist for treatment planning.
                - Engage in physical therapy and follow prescribed medication.
                - Explore speech and occupational therapy if needed.
                - Maintain a healthy diet and stay active as much as possible.
                """)
            else:
                st.success("✅ The person does **not** have Parkinson’s Disease")
                st.markdown("""
                <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
                <script>
                confetti({
                  particleCount: 150,
                  spread: 70,
                  origin: { y: 0.6 }
                });
                </script>
                """, unsafe_allow_html=True)
                st.info("""
                ### 🧘 General Health Tips:
                - Maintain physical and mental wellness.
                - Stay socially active and exercise regularly.
                - Stay hydrated and follow a good sleep routine.
                """)
                st.balloons()

from database import insert_calories
def calories_burnt_prediction_page():
    st.markdown("""
    <div style='background-color:#f9f9f9; padding:20px; border-radius:10px'>
        <h2 style='color:#fb8500;'>🔥 Calories Burnt Prediction</h2>
        <p style='font-size:16px;'>Estimate the amount of calories burned during exercise based on input metrics.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📝 Enter Workout Details", unsafe_allow_html=True)

    with st.form(key="calories_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox('⚥ Gender *', options=["Male", "Female"])
            weight = st.text_input('⚖️ Weight (kg) *')
            body_temp = st.text_input('🌡️ Body Temperature (°C) *')
        with col2:
            age = st.text_input('🎂 Age *')
            duration = st.text_input('⏱️ Workout Duration (min) *')
        with col3:
            height = st.text_input('📏 Height (cm) *')
            heart_rate = st.text_input('💓 Heart Rate *')

        submitted = st.form_submit_button("🔍 Predict Calories")

    if submitted:
        # Validate required fields
        fields = [gender, age, height, weight, duration, heart_rate, body_temp]
        
        if any(field.strip() == "" for field in fields):
            st.error("🚨 Please fill in **all required fields** to proceed.")
            return

        try:
            input_data = np.array([[safe_float(duration), safe_float(heart_rate), safe_float(body_temp),
                        safe_float(age), safe_float(height), safe_float(weight),
                        safe_float(gender)]])  # gender should be 1 for Male, 0 for Female

        except ValueError:
            st.error("❌ Invalid input! Please ensure all fields contain numbers.")
            return

        calories_prediction = models['calories'].predict(input_data)
        calories_value = calories_prediction[0]
         # ✅ Insert into database
        insert_calories(input_data.flatten().tolist(), calories_value)
        st.divider()
        st.success(f"🔥 Estimated Calories Burnt: {calories_value:.2f} kcal")

        if calories_value < 200:
            st.info("""
            ### 💡 Suggestions:
            - Increase workout duration or intensity.
            - Incorporate cardio or HIIT.
            - Track your daily calorie goal.
            """)
        elif 200 <= calories_value <= 500:
            st.info("""
            ### 👍 Good Job!
            - Maintain your current routine.
            - Stay hydrated and eat balanced meals.
            - Monitor fitness progress weekly.
            """)
            st.balloons()
        else:
            st.info("""
            ### 🏆 Great Performance!
            - Add rest/recovery days.
            - Replenish with protein & complex carbs.
            - Stretch post-exercise.
            """)
            st.snow()
            

# Route to selected page
if selected =='Home':
    home_page()
elif selected == 'Diabetes Prediction':
    diabetes_prediction_page()
elif selected == 'Heart Disease Prediction':
    heart_disease_prediction_page()
elif selected == 'Parkinsons Prediction':
    parkinsons_prediction_page()
elif selected == 'Calories Burnt Prediction':
    calories_burnt_prediction_page()
elif selected == "Dashboard":
    dashboard_page()
elif selected == "Data Visualiser":
    data_visualiser_page()
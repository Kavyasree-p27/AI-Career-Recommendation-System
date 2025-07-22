import streamlit as st 
import joblib
import numpy as np

model=joblib.load('models/career_model.pkl')
encoder=joblib.load('models/encoders.pkl')
career_encoder=encoder['career_aspiration']

career_descriptions = {
    "Software Engineer": "Designs and builds software systems and applications.",
    "Banker": "Handles financial transactions and services in banking sector.",
    "Social Network Studies": "Studies how people interact and behave on social platforms.",
    "Doctor": "Medical professionals who diagnose and treat illnesses.",
    "Engineer": "Problem-solvers who design systems, machines, and structures.",
    "Data Scientist": "Experts in extracting insights from data using AI/ML.",
    "Teacher": "Educators who guide learning and knowledge development.",
    "Researcher": "Inquisitive minds who discover new knowledge.",
    "Lawyer": "Legal professionals who interpret and apply the law."
}

input_order = ['gender', 'part_time_job', 'extracurricular_activities', 'absence_days',
               'weekly_self_study_hours', 'math_score', 'history_score', 'physics_score',
               'chemistry_score', 'biology_score', 'english_score', 'geography_score']

st.title("AI career recommendation system")

#collecting user inputs
import streamlit as st

st.title("🎓 AI Career Recommendation System")

# --- 1. Input Section ---
gender = st.selectbox("Gender", ["Select..", "female", "male"])
part_time_job = st.selectbox("Do you have a part-time job?", ["Select..", "True", "False"])
extracurricular = st.selectbox("Did you participate in extracurricular activities?", ["Select..", "True", "False"])
absence_days = st.slider("Number of absence days", -1, 30, -1)
weekly_hours = st.slider("Weekly self-study hours", -1, 40, -1)

# Subject scores
math = st.slider("Math Score", -1, 100, -1)
history = st.slider("History Score", -1, 100, -1)
physics = st.slider("Physics Score", -1, 100, -1)
chemistry = st.slider("Chemistry Score", -1, 100, -1)
biology = st.slider("Biology Score", -1, 100, -1)
english = st.slider("English Score", -1, 100, -1)
geography = st.slider("Geography Score", -1, 100, -1)

# --- 2. Validate only after button is clicked ---
if st.button("🎯 Get Career Recommendations"):

    # Validate selectboxes
    if gender == "Select.." or part_time_job == "Select.." or extracurricular == "Select..":
        st.warning("⚠️ Please make valid selections for gender, job, and activities.")
        st.stop()

    # Validate sliders
    if -1 in [absence_days, weekly_hours, math, history, physics, chemistry, biology, english, geography]:
        st.warning("⚠️ Please fill in all numeric fields before proceeding.")
        st.stop()

    # If all inputs are valid, continue to prediction
    st.success("✅ Inputs valid! Proceeding to generate recommendations...")
    user_input={
        "gender":gender,
        "part_time_job":part_time_job,
        "extracurricular_activities":extracurricular,
        "absence_days":absence_days,
        "weekly_self_study_hours":weekly_hours,
        "math_score": math,
        "history_score": history,
        "physics_score": physics,
        "chemistry_score": chemistry,
        "biology_score": biology,
        "english_score": english,
        "geography_score": geography
    }
    
    for col in ['gender','part_time_job','extracurricular_activities']:
        le=encoder[col]
        user_input[col]=le.transform([user_input[col]])[0]

    X_new = np.array([[user_input[feature] for feature in input_order]])#creates a 2d array so the ml model can understand

    proba=model.predict_proba(X_new)[0] # predicts probability for each career 

    career_encoder=encoder['career_aspiration']
    top3_sug=np.argsort(proba)[-3:][::-1] #sort the probability of each career and then reverses it
    top3_careers=career_encoder.inverse_transform(top3_sug)

    st.success("🎯 Top 3 Career Recommendations:")
    for i, career in enumerate(top3_careers, start=1):
        st.markdown(f"**{i}. {career}**")
        st.write("📘", career_descriptions.get(career, "Description not available."))

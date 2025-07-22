import joblib
import numpy as np

model=joblib.load('models/career_model.pkl')
encoder=joblib.load('models/encoders.pkl')

#temp user input
user_input = {
    "gender": "male",
    "part_time_job": "True",
    "extracurricular_activities": "True",
    "absence_days": 3,
    "weekly_self_study_hours": 12,
    "math_score": 95,
    "history_score": 75,
    "physics_score": 80,
    "chemistry_score": 85,
    "biology_score": 70,
    "english_score": 65,
    "geography_score": 60
}

for col in ['gender','part_time_job','extracurricular_activities']:
    le=encoder[col]
    user_input[col]=le.transform([user_input[col]])[0]
    
#arrange the input order
input_order = ['gender', 'part_time_job', 'extracurricular_activities', 'absence_days',
               'weekly_self_study_hours', 'math_score', 'history_score', 'physics_score',
               'chemistry_score', 'biology_score', 'english_score', 'geography_score']
X_new = np.array([[user_input[feature] for feature in input_order]])#creates a 2d array so the ml model can understand

proba=model.predict_proba(X_new)[0] # predicts probability for each career 

career_encoder=encoder['career_aspiration']
top3_sug=np.argsort(proba)[-3:][::-1] #sort the probability of each career and then reverses it
top3_careers=career_encoder.inverse_transform(top3_sug)

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
print("Top career reccommendation:\n")
for i,career in enumerate(top3_careers,start=1):
    print(f"\n{i}.{career}")
    print(career_descriptions.get(career, "Description not available."))
    


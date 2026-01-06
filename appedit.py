import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb


try:
    model = xgb.XGBClassifier()
    model.load_model("xgb_model.json")
    imputer = joblib.load("imputer.pkl")
except FileNotFoundError:
    st.error("Model or imputer file not found. Please ensure 'xgb_model.json' and 'imputer.pkl' are in the correct directory.")
    st.stop()



FEATURES = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount', 
    'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel', 
    'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 
    'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 
    'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 
    'YearsSinceLastPromotion', 'YearsWithCurrManager', 'TenureRatio', 
    'PromotionGap', 'Gender_Male', 'Department_Research & Development', 
    'Department_Sales', 'JobRole_Human Resources', 'JobRole_Laboratory Technician', 
    'JobRole_Manager', 'JobRole_Manufacturing Director', 'JobRole_Research Director', 
    'JobRole_Research Scientist', 'JobRole_Sales Executive', 
    'JobRole_Sales Representative', 'MaritalStatus_Married', 'MaritalStatus_Single', 
    'EducationField_Life Sciences', 'EducationField_Marketing', 'EducationField_Medical', 
    'EducationField_Other', 'EducationField_Technical Degree', 
    'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely', 'OverTime_Yes'
]

# --- Page Configuration ---
st.set_page_config(page_title="📊 HR Attrition Predictor", layout="wide")

# --- UI Title ---
st.title("📊 HR Attrition Predictor")
st.write("Provide employee details below to predict the likelihood of attrition.")

# --- Build input form in columns for a better layout ---
inputs = {} # Dictionary to store all inputs

# Create two main columns
left_col, right_col = st.columns(2)

# --- Left Column ---
with left_col:
    st.subheader("Demographics & Background")
    inputs['Age'] = st.slider("Age", 18, 70, 35)
    inputs['DistanceFromHome'] = st.slider("Distance From Home (miles)", 1, 30, 5)
    inputs['NumCompaniesWorked'] = st.number_input("Number of Past Companies", min_value=0, max_value=20, value=2)

    # --- Mappings for Categorical Inputs ---
    gender_map = {"Male": 1, "Female": 0}
    user_gender = st.radio("Gender", list(gender_map.keys()))
    inputs['Gender_Male'] = gender_map[user_gender]
    
    marital_status_map = {"Single": "Single", "Married": "Married", "Divorced": "Other"}
    user_marital = st.selectbox("Marital Status", list(marital_status_map.keys()))
    inputs['MaritalStatus_Single'] = 1 if user_marital == "Single" else 0
    inputs['MaritalStatus_Married'] = 1 if user_marital == "Married" else 0

# --- Right Column ---
with right_col:
    st.subheader("Job & Compensation Details")
    inputs['MonthlyIncome'] = st.number_input("Monthly Income ($)", 1000, 20000, 5000, 100)
    inputs['PercentSalaryHike'] = st.slider("Percent Salary Hike", 10, 25, 15)
    inputs['TotalWorkingYears'] = st.slider("Total Working Years", 0, 50, 10)
    
    overtime_map = {"Yes": 1, "No": 0}
    user_overtime = st.radio("Works Overtime?", list(overtime_map.keys()))
    inputs['OverTime_Yes'] = overtime_map[user_overtime]

    travel_map = {"Travel Frequently": "Travel_Frequently", "Travel Rarely": "Travel_Rarely", "Non-Travel": "Other"}
    user_travel = st.selectbox("Business Travel Frequency", list(travel_map.keys()))
    inputs['BusinessTravel_Travel_Frequently'] = 1 if user_travel == "Travel Frequently" else 0
    inputs['BusinessTravel_Travel_Rarely'] = 1 if user_travel == "Travel Rarely" else 0

# You can add more inputs in expanders to keep the UI clean
with st.expander("Show More Details (Satisfaction, etc.)"):
    col1, col2, col3 = st.columns(3)
    with col1:
        inputs['JobSatisfaction'] = st.slider("Job Satisfaction", 1, 4, 3)
        inputs['EnvironmentSatisfaction'] = st.slider("Environment Satisfaction", 1, 4, 3)
    with col2:
        inputs['RelationshipSatisfaction'] = st.slider("Relationship Satisfaction", 1, 4, 3)
        inputs['WorkLifeBalance'] = st.slider("Work-Life Balance", 1, 4, 3)
    with col3:
        # Fill in other inputs with default values if not explicitly asked
        # This part is simplified; in a real app, you'd add more widgets
        for feat in FEATURES:
            if feat not in inputs:
                inputs[feat] = 0.0


# --- Prediction Button and Display ---
if st.button("🔮 Predict Attrition", type="primary"):
    # Create a DataFrame in the correct order
    df = pd.DataFrame([inputs])[FEATURES]
    
    # Impute the data
    df_imputed = imputer.transform(df)

    # Make predictions
    pred = model.predict(df_imputed)[0]
    prob = model.predict_proba(df_imputed)[0][1]

    st.subheader("Prediction Result")
    
    if pred == 1:
        st.error(f"High Risk of Attrition (Probability: {prob:.2%})")
        st.warning("Consider reviewing this employee's workload, compensation, and satisfaction levels.")
    else:
        st.success(f"Low Risk of Attrition (Probability: {prob:.2%})")
        st.info("This employee is likely to stay with the company.")
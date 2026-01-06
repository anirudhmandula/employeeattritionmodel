import pandas as pd
import joblib
import xgboost as xgb

# Load trained model and imputer
model = xgb.XGBClassifier()
model.load_model("xgb_model.json")
imputer = joblib.load("imputer.pkl")

# All features used during training
FEATURES = [
    'Age', 'DailyRate', 'DistanceFromHome',
    'Education', 'EmployeeCount', 'EnvironmentSatisfaction',
    'HourlyRate', 'JobInvolvement', 'JobLevel',
    'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
    'NumCompaniesWorked', 'PercentSalaryHike',
    'PerformanceRating', 'RelationshipSatisfaction',
    'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'TenureRatio', 'PromotionGap', 'Gender_Male',
    'Department_Research & Development', 'Department_Sales',
    'JobRole_Human Resources', 'JobRole_Laboratory Technician',
    'JobRole_Manager', 'JobRole_Manufacturing Director',
    'JobRole_Research Director', 'JobRole_Research Scientist',
    'JobRole_Sales Executive', 'JobRole_Sales Representative',
    'MaritalStatus_Married', 'MaritalStatus_Single',
    'EducationField_Life Sciences', 'EducationField_Marketing',
    'EducationField_Medical', 'EducationField_Other',
    'EducationField_Technical Degree',
    'BusinessTravel_Travel_Frequently',
    'BusinessTravel_Travel_Rarely',   # <-- fixed missing quote
    'OverTime_Yes'
]

# Collect values interactively
values = {}
for f in FEATURES:
    val = input(f"Enter {f}: ")
    try:
        values[f] = float(val)
    except ValueError:
        print(f"Invalid input for {f}, using 0.0")
        values[f] = 0.0

# Create DataFrame and apply imputer
df = pd.DataFrame([values])[FEATURES]
df_imputed = imputer.transform(df)

# Predict
pred = model.predict(df_imputed)[0]
prob = model.predict_proba(df_imputed)[0][1]

print("\nPrediction:", "Attrition" if pred == 1 else "No Attrition")
print("Confidence (probability of Attrition):", round(prob, 3))

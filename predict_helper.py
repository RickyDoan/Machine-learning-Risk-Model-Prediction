import pandas as pd
import numpy as np
import joblib
from joblib import load
# Load the pre-trained model and any associated data
save_model = load("artifact/save_model.joblib")
scaler = save_model['scaler']
best_model = save_model['best_model']
feature = save_model['feature']
col_to_scaler = save_model['col_to_scaler']

# list_model = {
#     'scaler' : scaler,
#     'col_to_scaler' : col_to_scaler,
#     'best_model' : best_model_optuna_logistic,
#     'feature' : feature
# }
def prepare_input_data(input_data):
    """
    Prepares the input data for prediction by matching columns for training.
    """
    # Create a DataFrame with zeros for all expected columns
    df_input = pd.DataFrame([{col: 0 for col in feature}])

    # Map and scale numerical features
    df_input['age'] = input_data['Age']
    df_input['number_of_open_accounts'] = input_data['Number of Open Accounts']
    df_input['credit_utilization_ratio'] = input_data['Credit Utilization (%)']
    df_input['loan_tenure_months'] = input_data['Loan Tenure (Months)']
    df_input['loan_to_income'] = input_data['Loan/Income Ratio']
    df_input['delinquent_to_loan'] = input_data['Delinquent/Loan Month (%)']
    df_input['avg_dpd_delinquent'] = input_data['Avg DPD/Delinquent Ratio']

    # Get full col to scaler :
    for col in col_to_scaler:
        if col not in df_input.columns:
            df_input[col] = 1  # Default value for missing columns
    #Scaler :
    df_input[col_to_scaler] = scaler.transform(df_input[col_to_scaler])

    # One-hot encode categorical features
    if input_data['Residence Type'] == "Owned":
        df_input['residence_type_Owned'] = 1
    elif input_data['Residence Type'] == "Rented":
        df_input['residence_type_Rented'] = 1

    if input_data['Loan Purpose'] == "Education":
        df_input['loan_purpose_Education'] = 1
    elif input_data['Loan Purpose'] == "Home":
        df_input['loan_purpose_Home'] = 1
    elif input_data['Loan Purpose'] == "Personal":
        df_input['loan_purpose_Personal'] = 1

    if input_data['Loan Type'] == "Unsecured":
        df_input['loan_type_Unsecured'] = 1

    # Get dataframe with feature belong to training model
    df = df_input[feature]
    return df

def prepare_training_model(df, base_score= 300, length_score = 600) :
    x = np.dot(df.values, best_model.coef_.T) + best_model.intercept_
    default_probabilities = 1/(1+np.exp(-x))
    non_default_probabilities = 1- default_probabilities
    credit_score = base_score +  (non_default_probabilities.flatten() * length_score)

    def rating(credit_score):
        if 300 < credit_score <= 500:
            return "Poor"
        elif 500 < credit_score <= 650:
            return "Average"
        elif 650 < credit_score <= 750:
            return "Good"
        elif 750 < credit_score <= 950:
            return "Excellent"
        else:
            return "Undefined"
    rating = rating(credit_score)
    return default_probabilities, credit_score, rating
    # rating:
    # 300 - 500 --> Poor
    # 500 - 650 --> Average
    # 650 - 750 --> Good
    # 750 - 950 --> Excellent

def predict(input_data):
    prepare_input = prepare_input_data(input_data)
    default_probabilities, credit_score, rating = prepare_training_model(prepare_input)
    return default_probabilities.flatten()[0], int(credit_score[0]), rating


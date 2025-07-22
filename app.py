from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import pickle
import os
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__, static_folder='.', template_folder='.')

Path("model").mkdir(parents=True, exist_ok=True)
Path("data").mkdir(parents=True, exist_ok=True)

EMPLOYMENT_TYPE_MAPPING = {
    'Private': 'full-time',
    'Self-emp-not-inc': 'contract',
    'Local-gov': 'full-time',
    'State-gov': 'full-time',
    'Self-emp-inc': 'freelance',
    'Federal-gov': 'full-time',
    '?': 'part-time'
}

JOB_TITLE_MAPPING = {
    'Machine-op-inspct': 'Big Data Engineer',
    'Farming-fishing': 'Analytics Engineer',
    'Protective-serv': 'AI Scientist',
    'Other-service': 'Applied Data Scientist',
    'Prof-specialty': 'AI Researcher',
    'Craft-repair': '3D Computer Vision Researcher',
    'Adm-clerical': 'BI Data Analyst',
    'Exec-managerial': 'Applied Machine Learning Scientist',
    'Tech-support': 'Big Data Architect',
    'Sales': 'Analytics Engineer',
    'Handlers-cleaners': 'Big Data Engineer',
    'Transport-moving': 'Big Data Engineer',
    '?': 'AI Researcher'
}

COUNTRY_LIST = ['SG', 'SI', 'Singapore', 'TN', 'TR', 'UA', 'US', 'United Kingdom', 'United States']

RELATIONSHIP_STATUS_MAPPING = {
    'Never-married': 'single',
    'Married-civ-spouse': 'married',
    'Divorced': 'divorced',
    'Widowed': 'widowed',
    'Separated': 'divorced',
    'Married-spouse-absent': 'divorced',
    'Married-AF-spouse': 'married'
}

def preprocess_data(df):
    df['Experience_Years'] = df['age'] - 18
    df['Experience_Years'] = df['Experience_Years'].clip(lower=0)
    df['Employment_Type'] = df['workclass'].map(EMPLOYMENT_TYPE_MAPPING).fillna('part-time')
    df['Job_Title'] = df['occupation'].map(JOB_TITLE_MAPPING).fillna('AI Researcher')
    df['Country'] = df['native-country'].where(df['native-country'].isin(COUNTRY_LIST), 'United States')
    df['Relationship_Status'] = df['marital-status'].map(RELATIONSHIP_STATUS_MAPPING).fillna('single')
    df['Salary'] = df['income'].map({'<=50K': 25000, '>50K': 75000})

    le_employment = LabelEncoder()
    le_job = LabelEncoder()
    le_country = LabelEncoder()
    le_relationship = LabelEncoder()
    df['Employment_Type'] = le_employment.fit_transform(df['Employment_Type'])
    df['Job_Title'] = le_job.fit_transform(df['Job_Title'])
    df['Country'] = le_country.fit_transform(df['Country'])
    df['Relationship_Status'] = le_relationship.fit_transform(df['Relationship_Status'])

    features = ['Experience_Years', 'Employment_Type', 'Job_Title', 'Country', 'Relationship_Status']
    return df[features], df['Salary'], {
        'le_employment': le_employment,
        'le_job': le_job,
        'le_country': le_country,
        'le_relationship': le_relationship
    }

def train_model():
    csv_path = 'data/adult-3-1.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError("CSV file not found. Please add data/adult-3-1.csv")
    df = pd.read_csv(csv_path)
    X, y, encoders = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    with open('model/salary_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'encoders': encoders}, f)

model_path = 'model/salary_model.pkl'
if not os.path.exists(model_path):
    train_model()

with open(model_path, 'rb') as f:
    loaded = pickle.load(f)
    model = loaded['model']
    encoders = loaded['encoders']

def predict_salary(features):
    experience = int(features.get('experience', 1))
    employment_type = encoders['le_employment'].transform([features.get('employment_type', 'full-time')])[0]
    job_title = encoders['le_job'].transform([features.get('job_title', 'AI Researcher')])[0]
    country = encoders['le_country'].transform([features.get('country', 'United States')])[0]
    relationship_status = encoders['le_relationship'].transform([features.get('relationship_status', 'single')])[0]
    input_data = np.array([[experience, employment_type, job_title, country, relationship_status]])
    return model.predict(input_data)[0]

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/style.css')
def serve_css():
    return send_from_directory('.', 'style.css', mimetype='text/css')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        predicted_salary = predict_salary(data)
        current_salary = data.get('current_salary', None)
        
        response_data = {
            'success': True, 
            'predicted_salary': int(predicted_salary)
        }
        
        if current_salary and current_salary > 0:
            response_data['current_salary'] = int(current_salary)
            response_data['difference'] = int(predicted_salary - current_salary)
            response_data['percentage_change'] = round(((predicted_salary - current_salary) / current_salary) * 100, 1)
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

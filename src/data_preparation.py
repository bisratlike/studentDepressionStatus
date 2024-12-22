import pandas as pd
from sklearn.preprocessing import StandardScaler , LabelEncoder
import numpy as np
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    df = df.drop(columns=["id","City",'Degree'])
    df = df.dropna()
    label_encoder = LabelEncoder()
    df["Gender"]=label_encoder.fit_transform(df["Gender"])
    df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].map({'No': 0, 'Yes': 1})
    df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map({'No': 0, 'Yes': 1})
    df["Dietary Habits"]=df["Dietary Habits"].map({'Healthy': 0, 'Unhealthy': 1})
    def convert_sleep_duration(duration):
        if "hours" in duration:
            if "Less than" in duration:
                return 0
            elif "More than" in duration:
                return 24
            else:
                parts = duration.split("-")
                if len(parts) == 2:
                    return (float(int(parts[0])) + float(int(parts[1][0]))) / 2
                else:
                    return float(int(parts[0]))
        return np.nan

    df["Sleep Duration"] = df["Sleep Duration"].apply(convert_sleep_duration)
    df["Sleep Duration"] = df["Sleep Duration"].fillna(df["Sleep Duration"].mean())
    numerical_columns = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 
                        'Study Satisfaction', 'Job Satisfaction', 
                        'Financial Stress']
    
    
    
    for col in numerical_columns:
        df[col] = df[col].fillna(df[col].median())
    
   
    
    scaler = StandardScaler()
    
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    df = df.dropna(subset = ["Depression"])
    df['Depression'] = df['Depression'].map({'Yes': 1, 'No': 0})
    return df, scaler

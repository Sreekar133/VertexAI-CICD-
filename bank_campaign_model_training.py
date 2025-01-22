import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from joblib import dump, load
from google.cloud import storage
import json
from google.cloud import bigquery
from datetime import datetime

# Set up Google Cloud Storage client
storage_client = storage.Client()
bucket = storage_client.bucket("sreekar_reddy")  # Replace with your actual bucket name

def load_data(path):
    """Loads the CSV data into a DataFrame."""
    return pd.read_csv(path, sep=";")

def encode_categorical(df, categorical_cols):
    """Encodes categorical columns using LabelEncoder."""
    le = LabelEncoder()
    df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
    return df

def preprocess_features(df):
    """Prepares feature matrix (X) and target vector (y)."""
    X = df.drop('y', axis=1)
    y = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

    # Scale the features
    sc = StandardScaler()
    X = pd.DataFrame(sc.fit_transform(X), columns=X.columns)
    return X, y

def bucket_pdays(pdays):
    """Buckets the 'pdays' feature."""
    if pdays == 999:
        return 0
    elif pdays <= 30:
        return 1
    else:
        return 2

def apply_bucketing(df):
    """Applies bucketing to 'pdays' and drops unnecessary columns."""
    df['pdays_bucketed'] = df['pdays'].apply(bucket_pdays)
    df = df.drop('pdays', axis=1)
    df = df.drop('duration', axis=1)
    return df

def train_model(model_name, X_train, y_train):
    """Trains the model based on the provided model name."""
    if model_name == 'xgboost':
        model = XGBClassifier(random_state=42)
    else:
        raise ValueError("Invalid model name.")

    # Train the model directly without using the pipeline
    model.fit(X_train, y_train)
    return model

def get_classification_report(model, X_test, y_test):
    """Generates a classification report for the given test set."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report

def save_model_artifact(model_name, model):
    """Saves the trained model as a joblib file and uploads it to GCS."""
    artifact_name = model_name + '_model.joblib'
    dump(model, artifact_name)
    # Upload model artifact to Google Cloud Storage
    model_artifact = bucket.blob('bank_campaign_artifact/' + artifact_name)
    model_artifact.upload_from_filename(artifact_name)

def write_metrics_to_bigquery(algo_name, training_time, model_metrics):
    """Writes model metrics to BigQuery."""
    client = bigquery.Client()
    table_id = "vertexai-project-447608.ml_ops.bank_campaign_model_metrics"
    table = bigquery.Table(table_id)

    row = {
        "algo_name": algo_name,
        "training_time": training_time.strftime('%Y-%m-%d %H:%M:%S'),
        "model_metrics": json.dumps(model_metrics)
    }
    errors = client.insert_rows_json(table, [row])

    if errors == []:
        print("Metrics inserted successfully into BigQuery.")
    else:
        print("Error inserting metrics into BigQuery:", errors)

def main():
    """Main function that orchestrates the pipeline."""
    input_data_path = "gs://sreekar_reddy/bank_campaign_data/bank-campaign-training-data.csv"  # Change as needed
    model_name = 'xgboost'
    
    # Load and preprocess the data
    df = load_data(input_data_path)
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    df = encode_categorical(df, categorical_cols)
    df = apply_bucketing(df)
    X, y = preprocess_features(df)
    
    # Handle imbalanced classes using RandomOverSampler
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # Train the model directly without using a pipeline
    model = train_model(model_name, X_train, y_train)
    
    # Get classification report for the model
    accuracy_metrics = get_classification_report(model, X_test, y_test)
    
    # Record the training time
    training_time = datetime.now()
    
    # Write metrics to BigQuery
    write_metrics_to_bigquery(model_name, training_time, accuracy_metrics)
    
    # Save the model artifact to Google Cloud Storage
    save_model_artifact(model_name, model)

if __name__ == "__main__":
    main()

import pandas as pd
import joblib
from flask import Flask, request, render_template, send_file
import os

app = Flask(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "doctor_attendance_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "dummy_npi_data.xlsx")

# Load model and encoder safely
model, encoder, df = None, None, None

def load_pickle(file_path, name):
    try:
        return joblib.load(file_path)
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None

model = load_pickle(MODEL_PATH, "model")
encoder = load_pickle(ENCODER_PATH, "encoder")

try:
    df = pd.read_excel(DATA_PATH)
except Exception as e:
    print(f"Error loading data file: {e}")

# Function to preprocess input
def preprocess_input(hour):
    if df is None:
        raise ValueError("Dataset not loaded.")

    df_filtered = df.copy()

    # Ensure 'Login Time' column exists
    if 'Login Time' not in df_filtered.columns:
        raise ValueError("Missing 'Login Time' column in dataset.")

    # Convert 'Login Time' to datetime
    df_filtered['Login Time'] = pd.to_datetime(df_filtered['Login Time'], errors='coerce')
    df_filtered['Hour'] = df_filtered['Login Time'].dt.hour  

    # Filter dataset for the given hour
    df_filtered = df_filtered[df_filtered['Hour'] == hour]

    if df_filtered.empty:
        return None  # No data available for this hour

    # Ensure required categorical columns exist
    categorical_columns = ['State', 'Region', 'Speciality']
    missing_cols = [col for col in categorical_columns if col not in df_filtered.columns]
    
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # Encode categorical variables using the same encoder
    if encoder is None:
        raise ValueError("Encoder not loaded.")

    try:
        encoded_data = encoder.transform(df_filtered[categorical_columns])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
    except Exception as e:
        raise ValueError(f"Encoding Error: {e}")

    # Merge encoded data and drop original categorical columns
    df_filtered = df_filtered.drop(columns=categorical_columns, errors='ignore').reset_index(drop=True)
    df_filtered = pd.concat([df_filtered, encoded_df], axis=1)

    # Drop unused columns before prediction
    X_new = df_filtered.drop(columns=['NPI', 'Login Time', 'Logout Time', 'Count of Survey Attempts'], errors='ignore')

    if X_new.empty:
        return None

    # Make predictions
    if model is None:
        raise ValueError("Model not loaded.")

    try:
        df_filtered['Prediction'] = model.predict(X_new)
    except Exception as e:
        raise ValueError(f"Prediction Error: {e}")

    # Select NPIs with high probability
    selected_npis = df_filtered[df_filtered['Prediction'] == 1][['NPI']]

    return selected_npis if not selected_npis.empty else None

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            hour = int(request.form["hour"])
            result_df = preprocess_input(hour)

            if result_df is None or result_df.empty:
                return render_template("home.html", message="⚠ No doctors found for this time. Try another hour.")

            # Ensure output directory exists
            output_dir = os.path.join(BASE_DIR, "output")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "doctors_list.csv")

            # Save result as CSV
            result_df.to_csv(output_path, index=False)

            return render_template("home.html", message="✅ Prediction complete. Download the result below.", download_link="/download")

        except Exception as e:
            return render_template("home.html", message=f"❌ Error: {str(e)}")

    return render_template("home.html", message="Enter an hour to find doctors.")

@app.route("/download")
def download():
    file_path = os.path.join(BASE_DIR, "output", "doctors_list.csv")
    if not os.path.exists(file_path):
        return render_template("home.html", message="⚠ No file found. Please predict first.")

    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

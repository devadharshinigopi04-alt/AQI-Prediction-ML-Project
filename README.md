# AQI Prediction and Pollution Level Classification Using Machine Learning

## Project overview
This application uses pollutant measurements (PM2.5, PM10, NO2, SO2, CO and O3) to:
1. Predict a numerical Air Quality Index (AQI) using a Random Forest Regressor.
2. Classify the pollution level using a Random Forest Classifier.
3. Display a short general health suggestion for the predicted category.

## Project structure
- `app.py` – Streamlit user interface
- `train_model.py` – model training and evaluation
- `data/aqi_dataset.csv` – illustrative sample dataset
- `models/` – trained models are created here
- `requirements.txt` – Python dependencies

## Requirements
- Python 3.9 or later
- VS Code or another Python editor
- Internet connection for the first-time package installation

## Run in VS Code (Windows)
1. Extract the ZIP file.
2. Open the extracted `AQI_Prediction_ML_Project` folder in VS Code.
3. Open Terminal > New Terminal.
4. Create a virtual environment:

   `python -m venv venv`

5. Activate it:

   `venv\\Scripts\\activate`

6. Install packages:

   `python -m pip install -r requirements.txt`

7. Train the models:

   `python train_model.py`

8. Start the web application:

   `python -m streamlit run app.py`

9. Open the local URL displayed in the terminal (usually `http://localhost:8501`).

## Dataset note — important for academic submission
The included CSV is a **synthetic demonstration dataset**, not measured city data and not an official CPCB dataset. Its AQI target is generated using an illustrative formula so that the complete project can run immediately. Do not present its results as real-world accuracy or official AQI performance.

For a research-quality project, replace `data/aqi_dataset.csv` with a verified dataset (for example, CPCB/official monitoring data or a documented public dataset). Keep the six feature column names and the `AQI` and `Pollution_Level` target columns. Retrain the models with `python train_model.py`.

## Pollution category bands used in this sample
- Good: 0–50
- Satisfactory: 51–100
- Moderate: 101–200
- Poor: 201–300
- Very Poor: 301–400
- Severe: 401–500

These bands follow the commonly used Indian AQI category ranges. Always verify the latest official methodology before a formal deployment.

## Algorithms
- Random Forest Regressor: predicts the numerical AQI.
- Random Forest Classifier: predicts the pollution category.

## Limitations
- Output quality depends on the quality, coverage and representativeness of the training data.
- This is an academic demonstration, not an official monitoring system or medical advice.
- Sensor calibration, location, timestamp and pollutant-specific official AQI calculations matter in real deployments.

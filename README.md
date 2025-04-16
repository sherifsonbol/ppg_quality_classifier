
# PPG Signal Quality Classifier

This project uses machine learning to classify photoplethysmogram (PPG) signals as either good or poor quality.

## Features
- Feature extraction from PPG signals
- Model training using XGBoost and SMOTE
- Interactive web interface using Streamlit

## Folders
- `data/` - CSV files (`ppg_features.csv`, `quality-hr-ann.csv`)
- `models/` - Trained model (`ppg_quality_model_xgboost.joblib`)
- `scripts/` - Training code
- `app/` - Streamlit web app

## How to Run
```bash
pip install -r requirements.txt
python scripts/train_with_xgboost.py
streamlit run app/ppg_webapp.py
```

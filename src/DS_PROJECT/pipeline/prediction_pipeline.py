import joblib
import numpy as np
import pandas as pd
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        # Column names matching the schema.yaml
        self.feature_names = [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ]

    def predict(self, data):
        # Convert numpy array to DataFrame with proper column names
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=self.feature_names)
        prediction = self.model.predict(data)
        
        return prediction[0] if len(prediction) == 1 else prediction

import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from mlflow.exceptions import RestException
import numpy as np
import joblib
import json
from src.DS_PROJECT.constants import *
from src.DS_PROJECT.utils.common import read_yaml,create_directories,save_json

from src.DS_PROJECT.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            predicted_qualities = model.predict(test_x)

            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
            
            # Saving metrics as local
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)


            # Log the model (handle DagsHub limitation with model registry endpoints)
            try:
                mlflow.sklearn.log_model(model, "model")
            except RestException as e:
                # DagsHub doesn't support the logged model endpoint used by newer MLflow versions
                # Fall back to logging model as a regular artifact
                import tempfile
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    model_path = os.path.join(tmpdir, "model.joblib")
                    joblib.dump(model, model_path)
                    # Log as artifact
                    mlflow.log_artifact(model_path, "model")
                print(f"Note: Model logged as artifact (backend limitation: unsupported endpoint)")
            except Exception as e:
                # Re-raise if it's a different error
                raise e
    

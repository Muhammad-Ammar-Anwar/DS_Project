from src.DS_PROJECT.pipeline.data_ingestion_pipeline import STAGE_NAME
from src.DS_PROJECT.config.configuration import ConfigurationManager
from src.DS_PROJECT.components.model_trainer import ModelTrainer
from src.DS_PROJECT import logger

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()


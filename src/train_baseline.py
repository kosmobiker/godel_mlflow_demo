import pandas as pd
import awswrangler as wr
import numpy as np
import logging

from typing import Tuple

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_validate


TRACKING_SERVER_HOST = "ec2-34-250-13-150.eu-west-1.compute.amazonaws.com"
PATH_DATA = 's3://test-bucket-vlad-godel/data/olx_house_price_Q122.csv'
SEED = 42
 
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
client = MlflowClient(tracking_uri=f"http://{TRACKING_SERVER_HOST}:5000")

class Trainer():
    """
    Realization of trainer to demonstrate Mlflow functions
    """
    def __init__(self, experiment_name, data_path):
        self.experiment_name = experiment_name
        self.data_path = data_path
        if self.experiment_name:
            mlflow.set_experiment(experiment_name)
        self.X_train_transformed, self.X_test_transformed, self.y_train, self.y_test = self.preprocess(data_path)


    def preprocess(self, data_path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """This function makes preprocessing of raw
        input data and return datafrmaes for further steps

        Args:
            path (str): Path to S3 URI with raw dataset

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: datasets after transformation and splitting 
        """
        logging.debug(f"Pandas version is {pd.__version__}")
        logging.debug(f"Scikit-learn version is {sklearn.__version__}")
        logging.debug(f"MLflow version is {mlflow.__version__}")
        df = wr.s3.read_csv([self.data_path], encoding='utf-8')
        logging.info(f"Get data from S3")
        categorical_features = ['offer_type', 'offer_type_of_building',
                            'market', 'voivodeship', 'month']
        numeric_features = ['floor', 'area', 'rooms', 'longitude', 'latitude']
        df = df[(df["price"] <= df["price"].quantile(0.95)) & (df["price"] >= df["price"].quantile(0.05))]
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='infrequent_if_exist', min_frequency=2000))
                ])
        numeric_transformer = Pipeline(steps=[
                ('imputer', IterativeImputer(initial_strategy='mean', max_iter=5, random_state=SEED, verbose=0)),
                ('scaler' , StandardScaler())
                    ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ])
        y = df["price"]
        X_train, X_test, y_train, y_test= train_test_split(df, y, test_size=0.2, random_state=SEED)
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        logging.info('Data preprocessing successfully finished')
        
        return X_train_transformed, X_test_transformed, y_train, y_test
    
    def train(self) -> Tuple[str, str]:
        """This function trains several models 
        and track them into MLflow

        Args:
            X_train_transformed (np.array): Train dataset
            y_train (np.array): Target variable

        Returns:
            Tuple[str, str]: id of exeriment and id of best run
        """
        logging.info(f"tracking URI: '{mlflow.get_tracking_uri()}'")
        models = (
            [Ridge, "Ridge"], 
            [RandomForestRegressor, "RF"], 
            [LinearSVR, "LinearSVR"], 
            [KNeighborsRegressor, "KNN"]
            )
        mlflow.sklearn.autolog()
        for model_class in models:
            with mlflow.start_run():
                mlflow.log_param("Train datset size", self.X_train_transformed.shape)
                mlflow.log_param("model", model_class[1])
                estimator = model_class[0]()
                cv_results = cross_validate(estimator,
                            self.X_train_transformed, self.y_train,
                            cv=5, n_jobs=-1,
                            scoring=('neg_mean_absolute_percentage_error',
                                    'neg_root_mean_squared_error'),
                            return_train_score=True)
                mean_test_mape = cv_results['test_neg_mean_absolute_percentage_error'].mean()
                mean_train_mape = cv_results['train_neg_mean_absolute_percentage_error'].mean()
                mean_test_rmse = cv_results['test_neg_root_mean_squared_error'].mean()
                mean_train_rmse = cv_results['train_neg_root_mean_squared_error'].mean()
                mlflow.log_metric("mean_test_mape", mean_test_mape)
                mlflow.log_metric("mean_train_mape", mean_train_mape)
                mlflow.log_metric("mean_test_rmse", mean_test_rmse)
                mlflow.log_metric("mean_train_rmse", mean_train_rmse)
                mlflow.sklearn.log_model(estimator, artifact_path="models")
                mlflow.end_run()
        

if __name__ == "__main__":
    trainer = Trainer("my-experiment-godel22", PATH_DATA)
    trainer.train()
    
            
            
        
        
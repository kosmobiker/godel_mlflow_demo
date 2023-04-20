"""This module contains class for hyperparameter optimization to demonstrate Mlflow functionality"""
import pandas as pd
import awswrangler as wr
import numpy as np
import logging
import time
import pickle
from typing import Tuple
import mlflow
from mlflow.tracking import MlflowClient
import xgboost as xgb
import optuna
from optuna.integration.mlflow import MLflowCallback
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error 


TRACKING_SERVER_HOST = "ec2-34-242-123-148.eu-west-1.compute.amazonaws.com"
EXPERIMENT_NAME = 'godel-cozy-ds-hyperopt'
DATA_PATH = 's3://test-bucket-vlad-godel/data/olx_house_price_Q122.csv'
MODEL_NAME = 'house_pricing_xgboost_model'
SEED = 42
N_TRIALS = 30
TIMEOUT = 3600


logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
client = MlflowClient(tracking_uri=f"http://{TRACKING_SERVER_HOST}:5000")
mlflc = MLflowCallback()

class HyperOpt():
    """
    Realization of hyperparameter optimization to demonstrate Mlflow functions
    """
    def __init__(self, experiment_name, data_path, model_name, n_trials, timeout):
        self.ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_name = f"{experiment_name}_{self.ts}"
        self.data_path = data_path
        self.model_name = model_name
        self.X_train_transformed, self.X_test_transformed, self.y_train, self.y_test = self.preprocess()
        self.dtrain = xgb.DMatrix(self.X_train_transformed, label=self.y_train)
        self.dtest = xgb.DMatrix(self.X_test_transformed)
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_trial = self.find_best_parameters()


    def preprocess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The `preprocess` method makes preprocessing of raw
        input data and return datafrmaes for further steps
        
        Args:
            path (str): Path to S3 URI with raw dataset
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: datasets after transformation and splitting 
        """
        logging.debug(f"Pandas version is {pd.__version__}")
        logging.debug(f"Scikit-learn version is {sklearn.__version__}")
        logging.debug(f"MLflow version is {mlflow.__version__}")
        logging.info(f"Get data from S3")
        df = wr.s3.read_csv([self.data_path], encoding='utf-8')
        categorical_features = ['offer_type', 'offer_type_of_building', 'market', 'voivodeship', 'month']
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
        with open('preprocessor.b', 'wb') as f_out:
            pickle.dump(preprocessor, f_out)
             
        return X_train_transformed, X_test_transformed, y_train, y_test
    
    
    def find_best_parameters(self) -> optuna.trial._frozen.FrozenTrial:
        """
        The `find_best_parameters` method searches for the best hyperparameters 
        for the XGBoost model using Optuna
        
        Returns:
            optuna.trial._frozen.FrozenTrial: contains the best hyperparameters found by Optun
        """
        
        @mlflc.track_in_mlflow()
        def objective(trial) -> float:
            """
            The objective function for an Optuna study that tunes hyperparameters for an XGBoost model using cross-validation.
            Returns the mean test MAPE (mean absolute percentage error) of the tuned model.
            
            Args:
                trial: An Optuna `Trial` object used to sample hyperparameters for the XGBoost model.
                
            Returns:
                The mean test MAPE of the tuned XGBoost model.
            """
            search_space = {
                'objective': 'reg:squarederror',
                'eval_metric': ['mape', 'rmse'],
                'booster': 'gbtree',
                'verbosity': 0,
                'eta': trial.suggest_float('eta', 0.001, 0.3),
                'gamma': trial.suggest_float('gamma', 0.001, 10, log=True),
                'max_depth': trial.suggest_int('max_depth', 5, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 10e-5, 10.0,  log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 10e-5, 10.0, log=True),
                }
            pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-mape")
            history = xgb.cv(search_space,
                            self.dtrain,
                            folds=RepeatedKFold(n_splits=4, n_repeats=2),
                            num_boost_round=300,
                            early_stopping_rounds=50,
                            seed=SEED,
                            callbacks=[pruning_callback])
            mlflow.log_param("Optuna_trial_num", trial.number)
            mlflow.log_params(search_space)
            mean_train_rmse = history["train-rmse-mean"].values[-1]
            mean_test_rmse = history["test-rmse-mean"].values[-1]
            mean_train_mape = history["train-mape-mean"].values[-1]
            mean_test_mape = history["test-mape-mean"].values[-1]
            mlflow.log_metric("mean_train_rmse", mean_train_rmse)
            mlflow.log_metric("mean_test_rmse", mean_test_rmse)
            mlflow.log_metric("mean_train_mape", mean_train_mape)
            mlflow.log_metric("mean_test_mape", mean_test_mape)
            return mean_test_mape
    
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
        study = optuna.create_study(
                        study_name=f"{self.experiment_name}",
                        pruner=pruner,
                        direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, callbacks=[mlflc])
        logging.info(f"Number of finished trials: {len(study.trials)}")
        mlflow.xgboost.autolog()
        best = study.best_trial
        logging.info(f"Best params: {best.params.items()}")
        
        return best
    
    
    def _metrics(self, true: np.array, pred:np.array) -> Tuple[float, float, float, float]:
        """Calculate metrics for model evaluation
        
        Args:
            true (np.array):true values
            pred (np.array): predicted values
            
        Returns:
            Tuple[float, float, float, float]: calculated metrics
        """
        mae = mean_absolute_error(true, pred)
        mape = mean_absolute_percentage_error(true, pred)
        rmse = mean_squared_error(true, pred, squared=False)
        r2 = r2_score(true, pred)
        return mae, mape, rmse, r2
    
    
    def evaluate_and_register(self):
        """
        The `evaluate_and_register` method runs the best XGBoost model based 
        on the hyperparameters obtained from `find_best_parameters`. 
        It evaluates the performance of the model on the train and 
        test set using Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), 
        Root Mean Squared Error (RMSE), and R-squared (R2). It then logs the model 
        in MLflow by registering it as a model artifact and updating the model version.
        """
        with mlflow.start_run():
            model = xgb.train(
                        dtrain=self.dtrain,
                        params=self.best_trial.params,
                        num_boost_round=300)
            y_pred_train = model.predict(self.dtrain)
            y_pred_test = model.predict(self.dtest)
            train_mae, train_mape, train_rmse, train_r2 = self._metrics(self.y_train, y_pred_train)
            test_mae, test_mape, test_rmse, test_r2 = self._metrics(self.y_test, y_pred_test)
            model_description = f"""
                    XGboost model with hyperparameters:
                    {self.best_trial.params}
                    Models metrics:
                    train_mae={train_mae}, train_mape={train_mape}, train_rmse={train_rmse}, train_r2={train_r2}
                    test_mae={test_mae}, test_mape={test_mape}, test_rmse={test_rmse}, test_r2={test_r2}         
                    """
            mlflow.xgboost.log_model(model, artifact_path="models_artifacts", registered_model_name=self.model_name)
            mlflow.log_artifact(local_path="preprocessor.b", artifact_path="models_pickle")
            client.update_model_version(
                    name=self.model_name,
                    version=client.get_latest_versions(self.model_name)[0].version,
                    description=model_description,
                    )
            mlflow.end_run()
                        
            
if __name__ == "__main__":
    hyperopt = HyperOpt(EXPERIMENT_NAME, DATA_PATH, MODEL_NAME, N_TRIALS, TIMEOUT)
    hyperopt.find_best_parameters()
    hyperopt.evaluate_and_register()        
    logging.info("Oh, We Happy")   
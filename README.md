# Godel MLflow demo

<img src="https://cdn.bulldogjob.com/system/companies/logos/000/002/343/original/godel_logo_colored.png"  width="600" height="200">

***A small demo for Cozy Godel DS Catchup***
___

## Agenda:

1. What is MLflow?
2. How to install MLflow and run it?
3. Optuna
4. Case study: House pricing predictions
    * Simple Trainer
    * Hyperparameter Optimization 

___
 
## What is MLflow?

*"An Open source platform for the machine learning lifecycle"*

<img src="https://mlflow.org/images/MLflow-logo-final-white-TM.png"  width="500" height="200">

[**Mlflow**](https://mlflow.org/docs/latest/index.html) is an open-source platform for the complete machine learning life cycle that allows data scientists to manage their end-to-end machine learning workflow. It provides tools for tracking experiments, packaging code into reproducible runs, and sharing and deploying models. Mlflow is designed to work with a variety of machine learning libraries and languages and supports popular storage and deployment platforms. Mlflow helps data scientists to keep track of their experiments, share their results with others, and reproduce their work in a systematic and organized way.

### MLflow Modules:

It's a Python package with four main modules:

- Tracking
- Models
- Model registry
- Projects (Out of scope of the catchup)

### Experiment tracking:

Keeping track of all the relevant information from an ML experiment; varies from experiment to experiment. Experiment tracking helps with Reproducibility, Organization and Optimization.

Tracking experiments in spreadsheets helps but falls short in all the key points.

Tracking experiments with MLflow:

- MLflow organizes experiments into runs and keeps track of any variables that may affect the model as well as its result; Such as: Parameters, Metrics, Metadata, the Model itself...
- MLflow also automatically logs extra information about each run such as: Source Code, Git Commit, Start and End time and Author.

### Models module:

MLflow's Models module is a component of the MLflow platform that provides a simple and consistent way of packaging machine learning models in multiple formats and deploying them to a variety of production environments. It enables you to track different versions of your models and their respective metrics, as well as to register, manage, and deploy models to production. MLflow Models supports various model formats including Python functions, PyTorch, TensorFlow, Scikit-Learn, and XGBoost, and allows you to deploy models in various ways, including locally, to a server, to a cloud platform like AWS or Azure, or to a container platform like Docker or Kubernetes.

### Model registry:

MLflow's Model Registry module is a centralized model store that allows teams to collaboratively manage and version control models throughout their entire lifecycle, from experimentation to production deployment. It provides a UI and API for model registration, versioning, sharing, and managing access control. Model Registry allows users to track model lineage and lineage artifacts, manage model approval and staging, and deploy models to production. It also provides the ability to create a model version history, to compare models, and to set up alerts for changes in model performance or metadata. Model Registry supports a range of machine learning frameworks and deployment tools, making it easier to integrate with your existing infrastructure.

## How to install MLflow and run it?
### Installing MLflow:

It can be easily installed using `pip` or `conda`:

`pip: pip install mlflow` or `conda: conda install -c conda-forge mlflow`

### MLFlow in practice

Exists different scenarios of working with ML models:

- Single data scientist, generally competition model (Kaggle or another)
- Cross functional team with one data scientist, in this exists only one model but someone in the company can check the model and how to recreate
- Multiple data scientist that work in multiple ml models, in this point is necessary get a control of deploys and how to is created every model

Examples of such scenarios are:

- MLFlow run in local

    + in this scenario is create a default folder with name `mlruns`, and in this is created the default experiment in folder 0 and only using yml meta
    + If another experiment is created this take the number 1 and save all data in artifacts
    + if run mlflow ui this load all experiments to show

- MLFlow using a local database and tracking server

    + generally is used when I need test multiple hyper parameters in the model
    + is necessary run mlflow server and add where is saved the artifacts
    + this created one folder `artifacts_local`, in this is saved the artifacts of model with metadata

- MLFlow remote

    + in this scenario generally is used when multiple data scientist work in multiple model
    + using a remote server (EC2 in case of AWS), S3 to save artifacts and RDS (PostgreSQL)
    + here is used the url of server
    + a good practice in this case is use ec2-role to access in S3, is not a good idea use a IAM user
    + another good practice is block the port and public access to RDS and use the EC2 to connect

More info about deployment of MLflow is [here](https://mlflow.org/docs/latest/tracking.html).


### Architecture overview

General scheme of our MLflow architecture:

<img src="https://mlflow.org/docs/latest/_images/scenario_4.png"  width="500" height="500">


Installation is not difficult. You can easily adopt it for your cloud infrastructure.

We will use:

- **EC2** to host a Tracking server, `t2.micro` is ok
- **S3** to store artifacts
- **AWS RDS** for backend, I use `t4g.micro` with Postgres - more than enough 

The diagram of out AWS infrastructure is on the picture below:

Some useful commands to install and setup the MLflow on EC2:
![Label](infrastructure//infra.drawio.svg)

```bash
sudo yum update
sudo yum -y install python-pip
pip3 install mlflow boto3 psycopg2-binary
aws configure
```

Command to start Tracking server 

```bash
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://DB_USER:DB_PASSWORD@DB_ENDPOINT:5432/DB_NAME --default-artifact-root s3://S3_BUCKET_NAME
```
More info is [here](https://github.com/kosmobiker/mlops-zoomcamp/blob/main/02-experiment-tracking/mlflow_on_aws.md).

Lifehacks:
* Do not forget to customize Security groups and add inbound rules!
* Check you IAM roles!
* Make sure you can access your S3 buckets from the EC2!
* Do not forget about ports: `5432` for Postgres and `5000` for MLflow server

### More about Mlflow

Some benefits in remote server are:

+ shared experiments
+ collaborate
+ give more visibility of the data

Some limitations of MLFlow:

+ does not exists Auth and Users
+ data versioning
+ model/data monitoring & alerting, this is outside of the scope

Some alternatives:

+ [Neptune](https://neptune.ai/)
+ [Comet](https://www.comet.com/site/)
+ [Weights & Biases](https://wandb.ai/site)
+ AWS SageMaker
+ many others...

## Optuna

Optuna is an open-source hyperparameter optimization framework for Python. It provides a lightweight interface for defining and running hyperparameter optimization tasks, including support for various state-of-the-art algorithms for sampling and pruning hyperparameter search spaces. With Optuna, you can easily optimize the hyperparameters of a machine learning model, neural network architecture, or any other type of model that requires tuning of multiple parameters. Optuna also provides a powerful visualization toolkit for analyzing the results of hyperparameter optimization experiments.

### How Optuna works?

Optuna uses an algorithm called Tree-structured Parzen Estimator (TPE) to choose hyperparameters. TPE is a type of Bayesian optimization algorithm that models the relationship between hyperparameters and the objective function being optimized. It maintains two probability distributions: one for the hyperparameters that have been suggested so far and one for the objective function evaluated at those hyperparameters. Based on these probability distributions, TPE decides which hyperparameters to try next. This process continues until the maximum number of trials is reached or until the algorithm converges to a solution. By using TPE, Optuna is able to efficiently explore the hyperparameter space and find good configurations in a relatively small number of trials.

In Optuna, a pruner is a mechanism that decides whether to stop a trial (i.e., terminate the training of a specific set of hyperparameters) earlier than the maximum number of iterations (or the maximum training time) specified. Pruning can help speed up the hyperparameter search by avoiding wasting computational resources on unpromising hyperparameters.

### Types of pruning
Optuna has several built-in pruning algorithms, such as `MedianPruner`, `PercentilePruner`, and `SuccessiveHalvingPruner`, among others. These pruning algorithms work differently and decide when to prune based on various criteria. For example, `MedianPruner` prunes if the current trial's intermediate value is below the median of previously reported values at that step. The `PercentilePruner` prunes if the current trial's intermediate value is below the defined percentile of previously reported values. The `SuccessiveHalvingPruner` prunes the trials with the lowest intermediate values after each successive halving.

Overall, the decision to prune a trial is based on a tradeoff between the cost of continuing the trial and the potential reward (i.e., improving the model's performance) if the trial continues. Pruning allows the algorithm to quickly discard unpromising hyperparameters and focus more resources on the more promising ones.


## Case study: House pricing predictions

***Poland OLX House Price Q1 2022***

60,000+ house prices in over 600+ cities in Poland. Analyses of the pricing and brand based on OLX Portal

Data is available [here](https://www.kaggle.com/datasets/g1llar/poland-olx-house-price-q122).

Features description:
- `offer_title`: offer title
- `price`: price in PLN
- `price_per_meter`: price in PLN for square meter
- `offer_type`: as value name
- `floor`: floor number for -1 --> basement, 0 --> Ground Floor, 10 --> floor 10+, 11 --> attic
- `area`: area in square meters
- `rooms`: number of rooms for 4 --> rooms 4+
- `offer_type_of_building`: as value name
- `market`: as value name
- `city_name`: name of city where home is
- `voivodeship`: name of voivodeship where home is
- `month`: data download month
- `year`: data download year
- `population`: city population where home is
- `longitude` and `latitude`: city coord

Dataset contains 62818 records about house pricing in Poland in Q1 2022 with 15 features. Our aim is to predict house price based on other features. We will use `MLflow` to track our experiments. 

In the first experiment, we will try to train several models, track them and compare results.
It will be a simple `Scikitlearn` pipeline which will take our data and train 4 different models. We will compare metrics (**RMSE** and **MAPE**) on cross validation and choose the best one.

In the second experiment, we will try to optimize hyperparameters of the best model using `Optuna` and `MLflow`. We will use `XGboost` model to achieve better results. Our goal is to run `XGBoost` cross validating, find best hyperparameters, track the results, register best model and use it for predictions on holdout dataset.



# Godel MLflow demo

<img src="https://cdn.bulldogjob.com/system/companies/logos/000/002/343/original/godel_logo_colored.png"  width="500" height="150">

Small demo for Cozy Godel DS catchups

## Agenda:

1. What is MLflow
2. How to install MLflow and run it?
3. Case study: House pricing predictions
4. 


## Experiment tracking:

Keeping track of all the relevant information from an ML experiment; varies from experiment to experiment. Experiment tracking helps with Reproducibility, Organization and Optimization

Tracking experiments in spreadsheets helps but falls short in all the key points.

## MLflow:

*"An Open source platform for the machine learning lifecycle"*

<img src="https://mlflow.org/images/MLflow-logo-final-white-TM.png"  width="500" height="200">


It's a Python package with four main modules:

- Tracking
- Models
- Model registry
- Projects (Out of scope of the course)

Tracking experiments with MLflow:

 - MLflow organizes experiments into runs and keeps track of any variables that may affect the model as well as its result; Such as: Parameters, Metrics, Metadata, the Model itself...
- MLflow also automatically logs extra information about each run such as: Source Code, Git Commit, Start and End time and Author.

## Installing MLflow:
```bash
pip: pip install mlflow
```
or
```bash
conda: conda install -c conda-forge mlflow
```

## MLFlow in practice

Exists differents scenaries with work with ml models:

- Single data scientist, generally competition model (kaggle or another)
- Cross functional team with one data scientist, in this exists only one model but someone in the company can check the model and how to recreate
- multiple data scientist that work in multiple ml models, in this point is necessary get a control of deploys and how to is created every model

A example of this scenaries is:

- MLFlow run in local

    + in this scenario is create a default folder with name mlruns, and in this is created the default experiment in folder 0 and only using yml meta
    + If another experiment is created this take the number 1 and save all data in artifacts
    + if run mlflow ui this load all experiments to show

- MLFlow using a local database and tracking server

    + generraly is used when I need test multiple hyper parameters in the model
    + is necesary run mlflow server and add where is saved the artifacts
    + this created one folder artifacts_local, in this is saved the artifacts of model with metadata

- MLFlow remote

    + in this scenario generally is used when multiple data scientist work in multiple model
    + using a remote server (EC2 in case of AWS), S3 to save artifacts and RDS (postgresql)
    + here is used the url of server
    + a good practice in this case is use ec2-role to access in S3, is not a good idea use a IAM user
    + another good practice is block the port and public access to RDS and use the EC2 to connect

More info about deployment of MLflow is [here](https://mlflow.org/docs/latest/tracking.html).

- Some benefits in remote server are:

    + shared experiments
    + collaborate
    + give more visibility of the data

Some limitations of MLFlow:

+ does not exists Auth and Users
+ data versioning
+ model/data monitoring & alerting, this is outside of the scope

Some alternatives:

+ Neptune
+ comet
+ weights & biases

## Case study

to be continued
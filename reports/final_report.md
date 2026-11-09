# Final Report - End-to-End Data Management Pipeline for a Recommendation System

GitHub repository: https://github.com/govindsatwani/recommendation-pipeline

Name: Govind Satwani  
Roll No.: 2025AG05056  
Course: Data Management for Machine Learning

## 1. Problem Statement

The goal of this assignment was to design and implement an end-to-end data management pipeline for a recommendation system.  
The aim was to build a complete and orchestrated pipeline that covers all required stages: data ingestion, validation, preparation, transformation, feature management, model training, evaluation, and orchestration.  
For the project, I have used the Amazon Reviews 2018 - Software (5-core) dataset from the McAuley Lab, which contains product reviews and metadata for software-related products.  
I selected this dataset because it contains user-item interactions, item attributes, and review text that can be transformed into useful features for ranking and personalization.   
The goal was to recommend relevant products to users based on their past review behavior and item metadata, and at the same time ensuring the data pipeline remains robust, clean, and reproducible.

## 2. Pipeline Architecture and Workflow

The pipeline has the following stages:
1. Ingestion
2. Validation
3. Preparation
4. Transformation
5. Feature management
6. Model training
7. Model evaluation
8. Recommendation generation

The pipeline is orchestrated with the help of Prefect. The complete orchestration is handled in `dags/recsys_flow.py`.  
This script provides the complete end-to-end execution path and ensures that each stage runs in the correct dependency order.  
To make the pipeline robust and modularised separate modules for each stage of the pipeline have been created, making maintenance and troubleshooting easier.  
The system trains two models:
1. A non-personalized popularity baseline
2. A personalized SVD-based recommender

## 3. Data Ingestion

The ingestion stage loads the raw data files and converts them into 3 tables:
1. users
2. items
3. interactions

The project uses the public Amazon Software dataset and stores the raw files under `data/raw/`.  
An ingestion manifest is also generated in `data/raw/ingestion_manifest.json`, which records file-level and table-level metadata including row counts, columns, file sizes, and SHA-256 hashes.  

## 4. Data validation and quality checks

The validation stage is a central part of the pipeline. It includes schema and other checks ensures that the data is valid before it is used for modeling.  
The validation layer checks for the following:
1. missing values
2. duplicate rows
3. incorrect data types
4. out-of-range ratings
5. invalid timestamps
6. referential integrity between interactions, users, and items

Rejected rows are written into `data/errors/` and logged with explicit reasons, while valid records are retained in processed Parquet tables.  
A validation report is also generated for each execution.  
In addition to the real dataset validation, the project includes a fault-injection validation workflow that deliberately corrupts a copy of the inputs to demonstrate that all validation rules behave correctly.  

## 5. Data Preparation and Transformation

Once the data passes validation, the preparation stage cleans and standardizes the records. This includes:
1. handling missing values
2. correcting price-related issues
3. standardizing text values
4. removing duplicate user-item interactions and keeping the latest valid interaction
5. creating train/test splits based on time

The data is split in a time-based manner rather than randomly to avoid look-ahead leakage.  
This is important in recommendation systems because a random split can allow future interactions to leak into model training.  
A time-based split also better simulates real deployment conditions, where the system recommends items to users based on past behavior and evaluates future interactions.  

The transformation stage then converts the prepared data into useful summary structures:
1. user activity summaries
2. item popularity summaries
3. sparse user-item interaction matrix
4. index mappings for users and items

## 6. Feature Management

A key part of the project is the custom feature layer. Features are organized into:
1. user features
2. item features
3. interaction features

The feature layer is reusable and is intended to support both training-time and serving-time use cases.  
The feature layer is important because it makes the pipeline more maintainable and prevents feature logic from being duplicated or drifted across modules.

## 7. Model Training

Two recommendation models have been trained and compared:

### 7.1 Popularity baseline
* This model ranks items based on how frequently they were interacted with in the training data. 
* It is a straightforward baseline and provides a useful lower bound for recommendation quality.

### 7.2 Personalized SVD recommender
* This model uses matrix factorization to learn latent user-item representations. 
* It captures individual user preferences and provides a personalized ranking signal, which is more aligned with recommendation tasks than a global popularity score.  

The models are trained with deterministic settings and their runs are logged using MLflow. This allows experiment tracking for parameters, metrics, run metadata, and model artifacts.   
The following details are logged:
1. model name
2. hyperparameters
3. training time
4. data provenance
5. evaluation metrics
6. artifacts  

This makes the project reproducible and easier to compare across model versions.

## 8. Model Evaluation

The evaluation stage compares both models using ranking metrics and rating metrics.

Ranking metrics:
1. Precision@10
2. Recall@10
3. NDCG@10
4. Coverage
5. Diversity

Rating metrics:
1. RMSE
2. MAE

The evaluation is performed on a time-based holdout set and uses relevant interactions defined as ratings greater than or equal to 4.  
Items already seen in training are excluded from the evaluation list to avoid trivial memorisation.  
The results show that the personalized SVD model outperforms the popularity baseline in most ranking metrics, particularly in recall and NDCG, while also improving coverage and reducing RMSE and MAE.   
This demonstrates and proves that personalization adds meaningful value over a simple popularity-based recommendation.

## 9. Orchestration and Reproducibility

The end-to-end workflow is orchestrated using Prefect. The flow runs the full pipeline in dependency order and produces a clean execution trail.  
The orchestration layer ensures that:
1. all stages execute in sequence
2. failures are visible
3. logs are captured
4. runs are reproducible
5. the workflow can be scheduled

The pipeline is fully reproducible with a single command:
1. create a virtual environment
2. install dependencies
3. run the Prefect workflow

## 10. Serving and Recommendation use

The project also includes a recommendation service that can generate suggestions for:
1. a known user
2. a new user with a few given reviews
3. an unknown user with no prior history

The code can recommend items based on either learned user preferences or a popularity fallback when no history exists.

## 11. Limitations

The current version of the code has the following limitations:
1. the evaluation is offline rather than online
2. the dataset is sparse
3. some cold-start and cold-item challenges remain
4. only a baseline and one personalized model were trained
5. no large-scale hyperparameter tuning was performed

## 12. Conclusion

* In this project, I have created a complete and end-to-end recommendation pipeline covering the required stages of ingestion, validation, preparation, transformation, feature management, model training, evaluation, and orchestration. 
* The code provides a clean and modularised structure, reproducible execution, validation, and a clear comparison between a popularity baseline and a personalized SVD model.
* There are a few limitations in the current version of the code, but the code is still durable and consistent enough to carefully ingest, validate, and transform the data; train models on good, validated, and clean data; perform rigorous testing and evaluation of the model; and use the trained model to provide recommendations to users and also run on a schedule if needed.

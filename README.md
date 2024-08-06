# Singapore_resale_flat_price_prediction
Introduction : 
    Through this project a  machine learning model is constructed and  implemented it as a user-friendly online application in order to provide accurate predictions about the resale values of apartments in Singapore. 
    This prediction is based on past transactions of resale flats, and it aims to aid both future buyers and sellers in evaluating the worth of a flat after it has been previously resold. 
    Resale prices are influenced by a wide variety of criteria, including location, the kind of apartment, the total square footage, and the length of the lease. With these criterias the model will try to predict the price. 
Domain : Real Estate

Link1 : https://singapore-resale-flat-price-prediction1.onrender.com


Prerequisites
Python -- Programming Language
pandas -- Python Library for Data Visualization
numpy -- Fundamental Python package for scientific computing in Python
streamlit -- open-source Python framework
scikit-learn -- Machine Learning library for the Python programming language
Data Source
Link : https://beta.data.gov.sg/collections/189/view


Project Workflow
The following is a fundamental outline of the project:

The Resale Flat Prices dataset has five distinct CSV files, each representing a specific time period. The time periods are 1990 to 1999, 2000 to 2012, 2012 to 2014, 2015 to 2016, and 2017 onwards. All the five datasets are merged and used as a  unified dataset.

The data will be converted into a desired format for analysis, and required cleaning and pre-processing procedures will be done at this stage. Relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date will be extracted. Any additional features that may enhance prediction accuracy will also be created.

Used the decision tree regressor to accurate forecast of the continuous variable 'resale_price'.

Developed a Streamlit webpage that enables users to input values for each column and get the expected resale_price value for the flats in Singapore.

This project has been deployed in Rendor platform, the url is given above.


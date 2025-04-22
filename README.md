# DLBDSMTP01
This repositroy is for my project from the course DLBDSMTP01 - From model to production

The machine learning model was developed and trained in the Google Colab environment, while the API was written and deployed locally on my laptop. This was necessary because deploying an API on Google Colab and simulating incoming data simultaneously is challenging. Since Colab only runs one cell at a time, starting the API prevents any other tasks from running concurrently.

The repository contains the ml_model.py file, which defines the machine learning model, a CSV file used for training, the scaler.pkl and anomaly_model.pkl files for deploying the model within the API, the api.py file, which implements the API and a data_generation file which simulates continuous data flow to the api.

## equipment_anomaly_data.csv
The file equipment_anomaly_data.csv was found on kaggle (https://www.kaggle.com/datasets/dnkumars/industrial-equipment-monitoring-dataset). 
In the task description it says that temperature, humidity, and sound volume are good indicators for an anomaly in the production cycle.
Since the Kaggle dataset does not include a sound column but does provide a vibration column—which arguably serves as an even better indicator—I decided to use the vibration attribute instead.
The columns used in this project are temperature, vibration, humidity, and faulty. The faulty column serves as the target variable, where a value of 0 indicates no fault and 1 indicates a fault.

## ml_model.py
The machine learning model used in this project is a Random Forest Classifier, well-suited for supervised classification tasks like fault detection. It works by building multiple decision trees during training and combining their outputs to make accurate and robust predictions. By analyzing temperature, vibration, and humidity data, the model determines whether a produced component is likely to be faulty.
To integrate the model into the API, two .pkl files are used: anomaly_model.pkl, which contains the trained Random Forest model, and scaler.pkl, which holds the StandardScaler used to normalize input data. These files ensure that incoming sensor values are processed in the same way as during training, allowing the model to make consistent and reliable predictions in real time.

## api.py 
The API provides an endpoint that accepts JSON input with temperature, vibration, and humidity. It scales the input using a saved scaler and uses the trained ml model to predict if a component is faulty or not. The response includes two values: 

the prediction (0 or 1) and the probability_faulty score.


## data_generation.py
The data_generation file simulates a continuous stream of sensor data from the factory environment.

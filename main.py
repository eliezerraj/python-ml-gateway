import subprocess
import sys
import json
import os
import pandas as pd
import numpy as np
import boto3
import sklearn
import argparse
import math
from sagemaker import Session
from sagemaker import get_execution_role
from flask import Flask, jsonify, request

app = Flask(__name__)

SCALER_PATH = "s3://eliezerraj-908671954593-dataset/customer/scaler.joblib"
DATASET_PATH = "s3://eliezerraj-908671954593-dataset/customer/customer_encoded_data.csv"
ENDPOINT_NAME_1 = "kmeans-serverless-ep-customer-model-v3-2024-05-01-02-15-43"
ENDPOINT_NAME_2 = "xgboost-serverless-ep-fraud-model-v3-2024-04-23-00-41-40"
ENDPOINT_NAME_3 = "rcf-serverless-ep-payment-anomaly-model-v1-2024-05-06-13-40-27"

PORT = 5010
API_VERSION = "1.0"
POD_NAME = "py-ml-sagemaker.local"

client = boto3.client(  service_name="sagemaker",
                        region_name='us-east-2')
runtime = boto3.client( service_name="sagemaker-runtime",
                        region_name='us-east-2')

# -------- install libraries ------------------
def install(package):
    print("---- installing package : ", package)
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# --------- load the requirements -------------
def install_requirements():
    with open('./requirements.txt', 'r') as f:
        for line in f.readlines():
            install(line.strip())

# -------- scale dataset ------------------
def data_scale(df_new_customer_data):
    print("---- data_scale ----")

    df_final = pd.concat([df_new_customer_data, df_customer], axis=0)
    df_final_scaled = scaler_load.fit_transform(df_final).astype('float32')

    return df_final_scaled[0]

# --------- init  ---------------------------
def init():
    print("---- init ----")

    PORT = os.environ.get('PORT', 5010)
    API_VERSION = os.environ.get('API_VERSION','1.0')
    POD_NAME = os.environ.get('POD_NAME','py-ml-sagemaker.local')

    SCALER_PATH = "s3://eliezerraj-908671954593-dataset/customer/scaler.joblib"
    DATASET_PATH = "s3://eliezerraj-908671954593-dataset/customer/customer_encoded_data.csv"
    ENDPOINT_NAME_1 = "kmeans-serverless-ep-customer-model-v3-2024-05-01-02-15-43"
    ENDPOINT_NAME_2 = "xgboost-serverless-ep-fraud-model-v3-2024-04-23-00-41-40"
    ENDPOINT_NAME_3 = "rcf-serverless-ep-payment-anomaly-model-v1-2024-05-06-13-40-27"

# --------- route  ---------------------------
@app.route('/health', methods=['GET'])
def health():
    print("---- /health ----")
    res = "true"
    return res, 200

@app.route('/live', methods=['GET'])
def live():
    print("---- /live ----")
    res = "true"
    return res, 200

@app.route('/info', methods=['GET'])
def info():
    print("---- /info ----")

    res = {
        "info_pod": POD_NAME,
        "version": API_VERSION,
        "server": {
            "port": PORT
        },
        "scaler_path": SCALER_PATH,
        "dataset_path": DATASET_PATH,
        "endpoint_name_1": ENDPOINT_NAME_1,
        "endpoint_name_2": ENDPOINT_NAME_2,
        "endpoint_name_3": ENDPOINT_NAME_3
    }

    return res, 200

@app.route('/customer/classification', methods=['POST'])
def customer_classification():
    print("------ customer/classification ------")

    data = request.get_json()
    
    print("request data :", data)

    education_map = {'Uneducated': 0,
                     'Unknown': 0,
                     'High School': 1,
                     'College': 2,
                     'Graduate': 3,
                     'Post-Graduate': 4,
                     'Doctorate': 5}

    income_map = {'Unknown': 0,
                  'Less than $40K': 1,
                  '$40K - $60K':2,
                  '$60K - $80K': 3,
                  '$80K - $120K':4,
                  '$120K +': 5}

    df_new_customer_data = [{'Customer_Age': data['age'], 
                            'Dependent_count': data['dependent'], 
                            'Education_Level_Quality': data['education_level'], 
                            'Income_Category_Quality': data['income'], 
                            }]

    df_new_customer_data = pd.DataFrame(df_new_customer_data)
    print(df_new_customer_data)

    # fit the dataset
    result_scaled = data_scale(df_new_customer_data)
    print("result_scaled :", result_scaled)

    # -----------------------------------
    string_list = [str(element) for element in result_scaled.tolist()]
    delimiter = ", "
    result_string = delimiter.join(string_list)
    print(result_string)
    # -----------------------------------

    payload = bytes(result_string, 'utf-8')

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME_1,
        Body=payload,
        ContentType="text/csv",
    )

    y_predict = response["Body"].read().decode()
    print(y_predict)

    return y_predict, 200

@app.route('/payment/fraudPredict', methods=['POST'])
def payment_fraudPredict():
    print("------ payment/fraudPredict ------")

    data = request.get_json()
    
    print("request data :", data)

    # Distance
    p1 = [0, 0]
    p2 = [data['coord_x'], data['coord_y']]
    distance = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    
    if data['card_model'] == 'VIRTUAL':
        ohe_card_model_chip = 0
        ohe_card_model_virtual = 1
    else:
        ohe_card_model_chip = 1
        ohe_card_model_virtual = 0

    ohe_card_type = 1
    # -----------------------------------
    payload = str(distance) + ',' + str(ohe_card_model_chip) + ',' + str(ohe_card_model_virtual) + ',' + str(ohe_card_type)  + ',' + str(data['amount']) + ',' + str(data['tx_1d']) + ',' + str(data['avg_1d']) + ',' + str(data['tx_7d']) + ',' + str(data['avg_7d']) + ',' + str(data['tx_30d']) + ',' + str(data['avg_30d']) + ',' + str(data['time_btw_cc_tx'])

    print("payload :", payload)          
    content_type = 'text/csv'
    #payload = '9.0,23.0,7.0,90.0,4.0,365.0,17.0,263.0,28.0,238.0,97582.0'

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME_2,
        Body=payload,
        ContentType="text/csv",
    )

    y_predict = response["Body"].read().decode()
    print(y_predict)
    return y_predict, 200

@app.route('/payment/anomaly', methods=['POST'])
def payment_anomaly():
    print("------ /payment/anomaly ------")
    
    data = request.get_json()
    
    print("request data :", data)

    df_new_data = [{'amount': data['amount'], 
                    'tx_1d': data['tx_1d'], 
                    'avg_1d': data['avg_1d'], 
                    'tx_7d': data['tx_7d'],
                    'avg_7d': data['avg_7d'], 
                    'tx_30d': data['tx_30d'], 
                    'avg_30d': data['avg_30d'], 
                    'time_btw_cc_tx': data['time_btw_cc_tx'], 
                    }]

    content_type = 'text/csv'
    payload =  str(data['amount']) + ',' + str(data['tx_1d']) + ',' + str(data['avg_1d']) + ',' + str(data['tx_7d']) + ',' + str(data['avg_7d']) + ',' + str(data['tx_30d']) + ',' + str(data['avg_30d']) + ',' + str(data['time_btw_cc_tx'])
    #payload = '9.0,23.0,7.0,90.0,4.0,365.0,17.0,263.0,28.0,238.0,97582.0'

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME_3,
        Body=payload,
        ContentType="text/csv",
    )

    y_predict = response["Body"].read().decode()
    #print(type(y_predict))
    print(y_predict)
    res = json.loads(y_predict)
    return res['scores'][0], 200

# --------- main  -------------------------
if __name__ == '__main__':
    print("---- main ----")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--req")
    args = parser.parse_args()

    if args.req == 's':
        install_requirements()

    init()

    global df_customer 
    global scaler_load

    # Shows scikit-learn version
    print("sklearn version: ", sklearn.__version__)

    # Show enviroments
    print("==> POD_NAME: ", POD_NAME)
    print("==> PORT: ",     PORT)
    print("==> API_VERSION: ", API_VERSION)
    print("==> scaler_path: ", SCALER_PATH)
    print("==> dataset_path: ", DATASET_PATH)
    print("==> endpoint_name_1: ", ENDPOINT_NAME_1)
    print("==> endpoint_name_2: ", ENDPOINT_NAME_2)
    print("==> endpoint_name_3: ", ENDPOINT_NAME_3)

    # Load dataset
    df_customer = pd.read_csv(DATASET_PATH)
    print("==> Dataset loaded !!!")
    print(df_customer.head(5))

    # Load scaler model
    import s3fs
    import joblib

    fs = s3fs.S3FileSystem()
    scaler_path = SCALER_PATH

    with fs.open(scaler_path, 'rb') as f:
        scaler_load = joblib.load(f)
     print("==> Model loaded !!!")

    app.run(host='0.0.0.0',
            port=PORT)
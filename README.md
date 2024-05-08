# python-ml-gateway

POC for test purposes.

Service on charge to connect with sagemaker endpoints inference

## diagram

## Endpoints

+ GET /info

+ GET /header

+ POST /customer/classification

        {
            "age": 20,
            "dependent": 4,
            "education_level": 1,
            "income": 1
        }

+ POST /payment/fraudPredict

        {
            "card_number":"111.111.000.001",
            "terminal_name": "TERM-1",
            "coord_x": 89,
            "coord_y": 30,
            "card_type":"CREDIT",
            "card_model":"VIRTUAL",
            "mcc":"COMPUTE",
            "status":"OK",
            "currency":"BRL",
            "amount": 55.55,
            "payment_at":"2024-02-14T22:59:01.859507132-03:00",
            "tx_1d": 2,
            "avg_1d": 222.22,
            "tx_7d": 3,
            "avg_7d": 333.33,
            "tx_30d": 6,
            "avg_30d": 666.66,
            "time_btw_cc_tx": 77
        }

+ POST /payment/anomaly

        {
            "amount": 100.55,
            "tx_1d": 2,
            "avg_1d": 222.22,
            "tx_7d": 3,
            "avg_7d": 333.33,
            "tx_30d": 6,
            "avg_30d": 666.66,
            "time_btw_cc_tx": 77
        }
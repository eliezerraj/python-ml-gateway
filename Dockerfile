#docker build -t py-ml-gateway .

FROM python:3.10-slim

RUN pip install --upgrade pip

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

CMD ["python3", "./main.py"] 

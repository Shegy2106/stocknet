FROM python:latest

WORKDIR /myapp

RUN apt-get update
RUN apt-get install -y git

RUN mkdir stocknet-dataset
RUN git clone https://github.com/yumoxu/stocknet-dataset.git stocknet-dataset

COPY ../stock-importer /myapp


RUN pip install -r /myapp/requirements.txt

CMD ["python", "/myapp/stock_import.py"]



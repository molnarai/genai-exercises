FROM nvcr.io/nvidia/pytorch:24.01-py3
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt
WORKDIR /app
COPY . /app/

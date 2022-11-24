FROM python:3.9.6-slim-buster
WORKDIR /app
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
COPY . /app
CMD python classification_model_training.py
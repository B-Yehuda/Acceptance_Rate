FROM python:3.9.6
WORKDIR /app
COPY . /app
EXPOSE 5000
RUN pip install -r requirements.txt
# RUN pip install requests
CMD python app.py
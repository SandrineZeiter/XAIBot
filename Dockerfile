FROM python:3.10.10-buster

RUN apt-get update && \
    apt-get install -y libatlas-base-dev && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 5000

CMD [ "python", "main.py" ]
FROM python:3.9.7

RUN apt-get update -y && apt-get upgrade -y

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

# create dockerignote
COPY . .

CMD [ "python", "./main.py" ]

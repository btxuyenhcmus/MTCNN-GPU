FROM python:3.9.4

RUN apt-get update -y

WORKDIR /app

ADD . .

RUN pip3 install -r requirements.txt

EXPOSE 3000

CMD ["python3", "app.py"]
FROM python:3.10

WORKDIR lobster/

COPY . .

RUN pip install -r requirements.txt

CMD python project/main.py
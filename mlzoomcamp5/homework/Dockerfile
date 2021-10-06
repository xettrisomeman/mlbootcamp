FROM agrigorev/zoomcamp-model:3.8.12-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

COPY . /app/

CMD ["python", "main.py"]



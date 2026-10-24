FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m src.data_ingestion && python -m src.train
EXPOSE 5000
CMD ["python", "run_app.py"]

FROM python:3.13-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir numpy opencv-python pytest
CMD ["python", "filtri_test.py"]

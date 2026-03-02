FROM apache/airflow:2.7.0-python3.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy and install requirements
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Pre-install common packages
RUN pip install --no-cache-dir \
    scikit-surprise \
    mlflow \
    pandas \
    numpy \
    matplotlib

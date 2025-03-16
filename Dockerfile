FROM odsai/df25-baseline:1.0

COPY requirements.txt .
RUN pip install -r requirements.txt
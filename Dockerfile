FROM odsai/df25-baseline:1.0

COPY pyproject.toml .
COPY uv.lock .
RUN uv export --no-hashes --format requirements-txt > requirements.txt
RUN pip install -r requirements.txt
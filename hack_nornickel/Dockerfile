FROM python:3.11-slim


RUN apt-get update && apt-get install -y poppler-utils

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock* /app/

RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

COPY . /app

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["python", "./main.py"]
FROM python:3.11-slim

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-por \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Executa ambos os scripts redirecionando stdout e stderr para o log do container
CMD ["sh", "-c", "python -u script.py --monitor --intervalo 1 2>&1 & python -u api_rest.py"]

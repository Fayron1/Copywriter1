FROM python:3.12-slim

WORKDIR /app

# Системные зависимости для PyMuPDF и lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Пути для Docker
ENV BOOKS_PATH=/app/Books
ENV OUTPUT_PATH=/app/output

# Зависимости Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скрипты (копируем всё из папки scripts)
COPY . .

# Точка входа — python module
ENTRYPOINT ["python", "-m"]
CMD ["copywriter_kb.main", "--agent", "all"]

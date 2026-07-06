FROM python:3.12-bookworm

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m playwright install --with-deps chromium

COPY . .

EXPOSE 8501

CMD streamlit run app-combined.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true

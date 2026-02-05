FROM python:3.12-slim

ENV PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    cmake \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE $PORT

CMD ["sh", "-c", "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"]

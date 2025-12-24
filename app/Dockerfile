# Pythonのバージョンを指定
FROM python:3.10

# ビルドに必要なシステムライブラリをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 環境変数を設定
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY ./requirements.txt /app/
RUN pip install -r requirements.txt

COPY ./ /app/

RUN chmod +x /app/entrypoint.sh

WORKDIR /app

CMD ["/app/entrypoint.sh"]

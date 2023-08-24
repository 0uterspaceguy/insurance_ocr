FROM python:3.8-slim-buster as base
FROM base as builder
RUN mkdir /install
WORKDIR /install
RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install gcc build-essential -y
COPY libraries/requirements.txt /requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install nvidia-pyindex
RUN pip install --prefix=/install --no-warn-script-location -r /requirements.txt

FROM base
RUN apt-get update
RUN apt-get install libleptonica-dev tesseract-ocr libtesseract-dev python3-pil tesseract-ocr-eng tesseract-ocr-script-latn -y
COPY --from=builder /install /usr/local
WORKDIR /opt/app
COPY . .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

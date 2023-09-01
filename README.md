# Сервис распознавания страхового полиса

## Установка

Клоним:

```
git clone https://github.com/0uterspaceguy/insurance_ocr.git
cd insurance_ocr

```

Собираем контейнеры:

```bash
sudo docker build -t ocr_api .
sudo docker pull nvcr.io/nvidia/tritonserver:21.12-py3
```

Поднимаем сервис:

```
sudo docker compose up 
```

## Использование

Пример отправки запроса:

```
curl -L -F  "file=@ex.png" http://0.0.0.0:5000/recognize/
```



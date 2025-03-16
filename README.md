# [Data Fusion 2025: Label Craft](https://ods.ai/competitions/data-fusion2025-labelcraft)

## Зависимости

Установить зависимости из lock
```commandline
pip install uv
uv sync
```

Добавить зависимость
```commandline
 uv add "scikit-learn~=1.6.1" 
```
Добавить dev зависимость
```commandline
 uv add --dev "ruff~=0.9.6" 
```

## Линтеры

Отформатировать код
```commandline
make pretty
```

Запустить линтеры
```commandline
make lint
```

## Докер

Базовый образ [`odsai/df25-baseline`](https://hub.docker.com/r/odsai/df25-baseline)

Запустить докер
```commandline
brew install colima
colima start --memory 10
```

Запулить образ
```commandline
docker pull alex1501/labelcraft-custom:1.0
```

Собрать образ
```commandline
docker build -t alex1501/labelcraft-custom:1.0 .
```

Запустить докер
```commandline
docker run -p 8000:8000 alex1501/labelcraft-custom:1.0
```

Запушить образ в докерхаб
```commandline
docker login
docker push alex1501/labelcraft-custom:1.0
```

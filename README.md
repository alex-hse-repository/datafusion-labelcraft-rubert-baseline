# [Data Fusion 2025: Label Craft](https://ods.ai/competitions/data-fusion2025-labelcraft)

## Краткое описание идеи

1. Будем предсказывать только листовые категории(если в на тесте не лист, хуже не будет)
2. Подготовка данных(см `data/dataset_v1`)
  - Удалим примеры с нелистовой разметкой(порядка 100k)
  - Отделяем валидацию следующим образом:
    - OOS(10%) -- если туда попала категория, то и все примеры из нее тоже
    - IS(10%) -- если туда попала категория, то примеры из нее должны быть и в train
3. Обучение модели(`notebooks/baseline.ipynb`):
  - Модель: `cointegrated/rubert-tiny2`
  - Label Smothing: разметка неидеально, поэтому поставим 0.8 на класс из разметки, остальные 0.2 размажем по соседним листам 

## Как обучить модель
- Положить данные соревнования в папку `data`
- Подготовить данные: `python data/dataset_v1/transform.py`
- Обучить модель: `python notebooks/Experiment.ipynb`

## Зависимости

Установить зависимости из lock
```commandline
pip install uv
uv sync
source .venv/bin/activate
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

# Датасет V1

## Предобработка

Submit:
- Удалим примеры с разметкой не до листа(порядка 100k)

Experiments:
- Удалим категории только с 1 примером

## Валидация
Отделим 20% валидационной выборки
- OOS(10%) -- если туда попала категория, то и все примеры из нее тоже
- IS(10%) -- если туда попала категория, то примеры из нее должны быть и в train
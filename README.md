# House Price Predictor — прогноз цены квартиры на вторичном рынке Москвы

Модель машинного обучения для автоматической оценки стоимости квартир
(вторичный рынок Москвы) и сервис для получения оценки по API.

**Итоговая модель:** CatBoostRegressor · **MAPE (test): 6,69 %** · **R² (test): 0,976**

---

## Команда и роли

| Участник | Роли | Зона ответственности |
|---|---|---|
| Ганибаев Равиль | ML-инженер, программист, тестировщик | EDA, признаки, обучение моделей, тестирование сервиса |
| Ерофеев Тимур | Scrum-мастер, программист, архитектор, технический писатель | Планирование, ТЗ, архитектура сервиса, документация |

Управление задачами: доска Striveapp 
Период работ: 01.09.2026 – 30.10.2026. 

## Задача

Регрессия: по 41 признаку объекта (площади, локация, дом, время публикации)
предсказать цену квартиры `price_rub` в рублях.
Бизнес-метрика — MAPE (средняя абсолютная процентная ошибка).

## Данные

- Источник: [Moscow Real Estate Sales and Rentals 2020–2026](https://www.kaggle.com/datasets/sergionefedov/moscow-real-estate-sales-and-rentals-20202026) (Kaggle).
- `secondary_market.csv` — 50 000 объявлений вторичного рынка, 26 столбцов, пропусков нет.
- `new_builds.csv` — 8 000 новостроек (использовался только для сравнения структуры).
- Разбиение: train 70 % / val 15 % / test 15 % (`random_state=42`).

## Маршрут исследования

1. Разведочный анализ данных (EDA) и контроль утечек.
2. Подготовка признаков: тренд и сезонность из даты, возраст дома, циклические признаки, `QuantileTransformer`, `log1p` для цели.
3. Сравнение базовых моделей: CatBoost, LightGBM, XGBoost, FlatNet.
4. Эксперименты с CatBoost (loss-функции, глубина, one-hot, Optuna).
5. Feature engineering: расстояния до landmarks (haversine), агрегаты по районам/станциям, взаимодействия → 41 признак.
6. Ансамбли: среднее, веса Optuna, стекинг Ridge на OOF-прогнозах.
7. Нейросети: Wide & Deep, Multi-Head Attention, TabNet.
8. Финальная модель, важность признаков, проверка на тесте, сохранение артефактов.

## Основные результаты (test, 7 500 объектов)

| Модель | MAPE, % | R² | MAE, руб |
|---|---|---|---|
| **CatBoost (финальная, 41 признак)** | **6,69** | **0,976** | **884 119** |
| LightGBM (val) | 7,22 | 0,971 | 944 329 |
| XGBoost (val) | 8,23 | 0,963 | 1 037 238 |
| FlatNet NN (val) | 8,17 | 0,966 | 1 036 322 |
| Стекинг Ridge (val) | 6,72 | — | — |

Выводы: градиентный бустинг превосходит нейросети на табличных данных объёма 50k;
ансамбли не дают значимого прироста (< 0,03 п.п.); удаление 21 «неважного» признака ухудшает test MAPE до 6,84 % → оставлены все 41.

## Воспроизведение

```bash
git clone <repo-url> && cd house-price-predictor
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebooks/house_price_predictor_full.ipynb
```

Сервис:

```bash
cd predictor-service
uvicorn app.main:app --reload
# POST /predict → {"predicted_price_rub": ...}
```

Артефакты обучения лежат в `artifacts/`: модель, трансформер, агрегаты train, метаданные.



## Цитирование источника данных

Sergey Nefedov. (2026). *Moscow Real Estate Sales and Rentals 2020–2026* [Data set]. Kaggle.
https://www.kaggle.com/datasets/sergionefedov/moscow-real-estate-sales-and-rentals-20202026

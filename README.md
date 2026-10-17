# House Price Predictor — Москва

Предсказание цены квартиры в Москве на основе 41 признака.

## Результаты

| Метрика | Test |
|---|---|
| MAPE | **6.69%** |
| R² | 0.9760 |
| MAE | 884 042 ₽ |
| RMSE | 1 327 745 ₽ |

## Модель

CatBoost с гиперпараметрами от Optuna:
- `learning_rate=0.029`, `depth=7`, `l2_leaf_reg=4.14`
- 3138 деревьев (early stopping)

## Структура
predictor-service/
├── notebooks/ # EDA, эксперименты, финальная модель
├── models/ # сохранённые модели (не в git)
├── data/ # датасеты (не в git)
└── figures/ # графики


## Установка

```bash
cd predictor-service
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r ../requirements.txt


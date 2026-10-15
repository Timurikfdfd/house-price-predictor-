# Прогноз цены квартиры на вторичном рынке Москвы

Блокнот `data_preprocessing.ipynb`, разбитый на 8 частей по его разделам.

## Части

| № | Блокнот | Что внутри | Главный результат |
|---|---|---|---|
| 1 | [01_eda.ipynb](01_eda.ipynb) | Загрузка данных и разведочный анализ | — |
| 2 | [02_features.ipynb](02_features.ipynb) | Подготовка признаков и нормализация | — |
| 3 | [03_boosting_baselines.ipynb](03_boosting_baselines.ipynb) | Первое сравнение моделей: CatBoost, LightGBM, XGBoost | CatBoost — MAPE 6,68 %, LightGBM — 7,15 %, XGBoost — 7,97 %, FlatNet — 8,17 %; среднее трёх бустингов — 6,79 %. |
| 4 | [04_catboost_experiments.ipynb](04_catboost_experiments.ipynb) | Эксперименты с CatBoost | лучший — базовый CatBoost с RMSE: MAPE 6,73 % на валидации; Huber не сошёлся. |
| 5 | [05_feature_engineering.ipynb](05_feature_engineering.ipynb) | Новые признаки и CatBoost на них | Групповые агрегаты теперь fit-ятся только на train; сохранённый ранее результат 6,665 % относится к предыдущему варианту и требует пересчёта. |
| 6 | [06_ensemble.ipynb](06_ensemble.ipynb) | Ансамбль CatBoost, LightGBM и XGBoost | CatBoost — 6,735 %, взвешенное среднее — 6,720 %, стекинг — 6,719 % MAPE на валидации. |
| 7 | [07_neural_networks.ipynb](07_neural_networks.ipynb) | Эксперименты с нейросетями | нейросети уступают бустингам: лучшая — базовая NN v1 (MAPE 8,17 %), Wide & Deep — 10,28 %, TabNet — 11,11 %, Attention — 12,16 %. |
| 8 | [08_final_model.ipynb](08_final_model.ipynb) | Важность признаков, проверка на тесте и финальная модель | Разбиение train/val/test — 70/15/15; групповые агрегаты вычисляются только по train. После запуска ячейки сохранения создаются `final_catboost.pkl`, `quantile_transformer.pkl`, `district_aggregates.json` и `final_metadata.json`. Выведенные ранее метрики требуют пересчёта. |

## Графики для отчёта

Лежат в папке `figures/`.

| Рисунок | Подпись | Файл | Часть |
|---|---|---|---|
| 1 | Распределения числовых признаков (среднее и медиана в заголовках) | [`fig_01_numeric_distributions.png`](figures/fig_01_numeric_distributions.png) | 1 |
| 2 | Распределения дискретных и категориальных признаков | [`fig_02_categorical_distributions.png`](figures/fig_02_categorical_distributions.png) | 1 |
| 3 | Матрица корреляций Пирсона для числовых признаков | [`fig_03_pearson_correlation.png`](figures/fig_03_pearson_correlation.png) | 1 |
| 4 | Медианная цена квартиры по месяцам публикации | [`fig_04_median_price_by_month.png`](figures/fig_04_median_price_by_month.png) | 2 |
| 5 | Корреляции Спирмена после кодирования признаков | [`fig_05_spearman_correlation.png`](figures/fig_05_spearman_correlation.png) | 2 |
| 6 | Сравнение архитектур нейросетей: ошибка на валидации по эпохам | [`fig_06_nn_validation_loss.png`](figures/fig_06_nn_validation_loss.png) | 7 |
| 7 | Важность признаков CatBoost (41 признак) | [`fig_07_catboost_feature_importance.png`](figures/fig_07_catboost_feature_importance.png) | 8 |

## Как пересчитать

Части — последовательные шаги одного исследования: код каждой части использует переменные из предыдущих.
Чтобы пересчитать, например, часть 3, выполните в её ядре по порядку `%run -i 01_eda.ipynb` и `%run -i 02_features.ipynb`,
а затем её собственные ячейки. Данные лежат рядом, в `Moscow_Real_Estate/`, поэтому пути из кода (`./Moscow_Real_Estate/...`)
работают из этой папки.

В `08_final_model.ipynb` `QuantileTransformer` сохраняется как исходный fitted-экземпляр; список его числовых колонок и обученные медианы пропусков записываются в `final_metadata.json`. JSON агрегатов содержит только train-группы и train-средние fallback для неизвестных районов/станций. Изменённые ячейки необходимо выполнить заново: ранее сохранённая модель и её метрики получены с агрегатами по всему датасету.

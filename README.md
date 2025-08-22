# Recommender System — MVP (просто и по делу)

## Что делаем
Делаем простой MVP рекомендательной системы для интернет‑магазина. Хотим увеличивать оборот за счёт релевантных товаров.
Главная тех. метрика в экспериментах: Precision@10 / Recall@10 для рекомендаций и AUC для модели покупки.

## Данные
Источник: Retail Rocket (`events`, `item_properties`, `category_tree`).
Объём: ~2.7M событий, ~1.4M пользователей, ~235k товаров.

## Признаки (features)
- События: `view`, `addtocart`, `transaction` (а также их счётчики по user–item).
- Время: день недели, час, «часть суток» (Night/Morning/Afternoon/Evening).
- Товары: топ‑20 свойств (включая `categoryid`, `available`) → one‑hot/счётчики.

## Валидация
Разбиваем по времени: train — до `2015‑07‑01`, test — после. Так не «подглядываем в будущее».

## Модели (кратко)
- **Baseline (Top‑N популярных)** → Precision@10 ≈ **0.0115**.  
- **LightFM (WARP)** → Precision@10 ≈ **0.0070**, Recall@10 ≈ **0.054**.  
- **XGBoost (классификация покупки)** → AUC ≈ **0.967**.  
Для MVP берём XGBoost: быстро, стабильно, даёт хороший AUC.

## REST API
Сервис принимает минимальные признаки и возвращает вероятность покупки.

### `POST /predict`
**Вход**:
```json
{"view": 3, "addtocart": 1}
```
**Ответ**:
```json
{"purchase_probability": 0.83}
```

### `GET /metrics`
Возвращает основную метрику качества модели (AUC):
```json
{"AUC": 0.9666}
```

### `GET /health`
Проверка, что сервис жив:
```json
{"status": "ok"}
```

## Как запустить

### В Docker
```bash
# локально собрать
docker build -t kotleha/recommender:latest .

# или стянуть готовый образ (если доступен в Docker Hub)
# docker pull kotleha/recommender:latest

# запустить
docker run --rm -p 8000:8000 kotleha/recommender:latest
```

### Локально (без Docker)
```bash
pip install -r requirements.txt
python app.py
# сервис будет на http://localhost:8000
```

### Примеры запросов (curl)
```bash
curl http://localhost:8000/health

curl http://localhost:8000/metrics

curl -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{"view": 3, "addtocart": 1}'
```

## Что ещё можно улучшить дальше
- Подмешать item/user‑факторы в модель (Category/Brand/Price bins, частоты, recency).
- Досчитать офлайн‑рекомендатели (ALS, item2vec) и сделать hybrid.
- Добавить ранжирование top‑N под слот главной страницы.
- Онлайн‑метрики и A/B — когда будет продовый трафик.

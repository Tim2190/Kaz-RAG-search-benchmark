# Как собрать корпус и положить его в GitHub

Шаги 1–2 запускаются **локально** (gov.kz и стеммер закрыты в облачном
окружении Claude). Результат — JSONL-файлы — коммитятся в репозиторий.

## 0. Установка

```bash
pip install datasets
```

## 1. Сбор корпуса из казахской Википедии  → `data/raw/wiki_kz.jsonl`

```bash
python -m src.scraping.wiki_scraper --target 800 --out data/raw/wiki_kz.jsonl
```

- По умолчанию (`--via hf`) берёт ГОТОВЫЙ дамп `wikimedia/wikipedia`
  с Hugging Face — чистый текст, БЕЗ лимитов API (стримит, весь не качает).
- Отсеивает короткие заготовки (`--min-len`) и не-казахские страницы.
- `--via api` — запасной путь через живой MediaWiki API (лимитируется
  на общих IP вроде Colab, поэтому по умолчанию выключен).

Одна строка = одна статья: `{id, title, source_name, project, url, text, published_at}`.

> Сборщики `news_scraper.py` (новостные сайты) и `gov_scraper.py` (gov.kz)
> оставлены как запасные. Новостные сайты дали мусор (парсер вытаскивал
> сайдбар вместо статей), поэтому базовый источник — Википедия.

## 2. Нарезка на пассажи  → `data/corpus/corpus.jsonl`

```bash
python -m src.corpus.build_corpus \
    --raw data/raw/wiki_kz.jsonl \
    --out data/corpus/corpus.jsonl \
    --target-words 120 --overlap-words 20
```

Чистит текст (выкидывает меню/боилерплейт) и режет на пассажи ~120 слов с
перекрытием. Это и есть единицы поиска для бенчмарка.

## 3. Коммит корпуса в GitHub

```bash
git add data/raw/wiki_kz.jsonl data/corpus/corpus.jsonl
git commit -m "Корпус: N статей / M пассажей (kk.wikipedia)"
git push origin claude/adoring-newton-GmiMr
```

После этого пришли мне число документов/пассажей — и переходим к Фазе 2
(генерация запросов и эталонов qrels).

## 4. Прогон бенчмарка «До/После» (Фаза 3–4)

**«До» — без стеммера** (работает где угодно, сети не требует):
```bash
python -m src.eval.run_benchmark --stemmer identity --out results/bm25_identity.json
```

**«После» — со стеммером** (нужен доступ к API стеммера; запускать в Colab):
```bash
python -m src.eval.run_benchmark --stemmer kazakh --out results/bm25_kazakh.json
```
- Перед индексацией прогревает кэш стеммера для ВСЕХ уникальных словоформ
  корпуса. Из-за лимита API (30 запросов/мин) это идёт ~1 час, НО:
  - прогрев устойчив к обрывам и докачивается из кэша при повторном запуске;
  - результат кэшируется в `data/resources/stem_cache.json` — закоммить его,
    и повторные прогоны будут мгновенными.

**Закоммитить кэш и результаты:**
```bash
git add data/resources/stem_cache.json results/
git commit -m "Стем-кэш + результаты бенчмарка До/После"
git push origin claude/adoring-newton-GmiMr
```

## 5. Dense-поиск: IBM Granite / Google LaBSE / e5 (Colab, GPU)

```bash
pip install sentence-transformers torch
python -m src.eval.run_dense --model labse   --out results/dense_labse_300.json
python -m src.eval.run_dense --model e5      --out results/dense_e5_300.json
python -m src.eval.run_dense --model granite --out results/dense_granite_300.json
```
Эмбеддинги корпуса кэшируются (`results/emb_<model>.npy`) — повтор не пересчитывает.
Сравнить с лексикой: `python -m src.eval.compare --before results/bm25_kazakh.json --after results/dense_labse_300.json`.

## 6. RAG-галлюцинации: IBM Granite / Google Gemma (Colab, GPU)

```bash
pip install transformers torch accelerate
python -m src.eval.run_rag --llm granite --top-k 3 --out results/rag_granite.json
python -m src.eval.run_rag --llm gemma   --top-k 3 --out results/rag_gemma.json
```
Прогоняет вопросы через LLM с контекстом от BM25 (без/со стеммером) и считает
долю correct / hallucination / abstain. Печатает дельту галлюцинаций «До→После».

---

### Проверка кода без сети
Метрики, токенизация, чанкинг, стем-клиент, dense и RAG-скоринг покрыты тестами:
```bash
python -m unittest discover tests      # 51 тест
```

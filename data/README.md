# Форматы данных

Все наборы — построчный JSONL (одна запись = одна строка JSON).

## `corpus/corpus.jsonl` — единицы поиска (пассажи)

```json
{
  "doc_id": "moa_0001_p3",
  "source_doc_id": "moa_0001",
  "source_name": "МинСельХоз",
  "title": "Тақырып...",
  "text": "Толық мәтін пассажы...",
  "lang": "kz",
  "url": "https://www.gov.kz/.../details/...?lang=kz",
  "published_at": "2026-05-20"
}
```

- `doc_id` — уникальный id пассажа (то, что ранжируем).
- `source_doc_id` — id исходного релиза (несколько пассажей на документ).

## `queries/queries.jsonl` — запросы

```json
{
  "query_id": "q_0007",
  "text": "Сұрақ мәтіні қазақша?",
  "category": "inflected",
  "gold_doc_ids": ["moa_0001_p3"],
  "generated_by": "llm",
  "validated": true
}
```

- `category` ∈ `inflected` | `vocabulary-gap` | `natural` — для атрибуции вклада
  стеммера vs словаря.
- `gold_doc_ids` — эталонные релевантные пассажи (qrels).
- `validated` — прошёл ли ручную проверку (Фаза 2).

## `queries/qrels.tsv` — релевантность (TREC-формат)

```
query_id  0  doc_id  relevance
```

`relevance` бинарная (1 = релевантен). Производится из `gold_doc_ids`.

## `resources/`

- `stemmer/` — казахский стеммер (код/правила), встраивается в `src/preprocess/stemmer.py`.
- `dictionary/` — толковый/синонимический словарь для расширения запроса (v2).

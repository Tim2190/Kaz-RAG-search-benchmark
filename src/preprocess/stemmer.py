"""
Точка встраивания КАЗАХСКОГО СТЕММЕРА.

Стеммер — ядро эксперимента: он обрезает агглютинативную морфологию,
сводя разные словоформы к общей основе, чтобы запрос и документ совпадали.

КАК ВСТРОИТЬ СВОЙ СТЕММЕР:
  1. Реализуй метод stem(token) -> str внутри класса KazakhStemmer ниже
     (или оберни свой готовый код).
  2. Больше нигде ничего менять не нужно — пайплайны (BM25 и пр.) берут
     стеммер через get_stemmer() и применяют к токенам и запроса, и корпуса
     ОДИНАКОВО (это критично: нормализация должна быть симметричной).

Пока метод не заполнен, используется IdentityStemmer (passthrough) —
это даёт корректный baseline «без стеммера» для сравнения До/После.
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from .tokenize import tokenize


@runtime_checkable
class Stemmer(Protocol):
    """Контракт стеммера: одна обязательная операция над токеном."""

    def stem(self, token: str) -> str:  # pragma: no cover - интерфейс
        ...


class IdentityStemmer:
    """Passthrough — основа для baseline «без морфологической нормализации»."""

    name = "identity"

    def stem(self, token: str) -> str:
        return token


# Казахский стеммер развёрнут как сервис (Cloud Run) и подключается через
# HTTP-клиент KazakhStemmerAPI (см. stemmer_api.py): батчинг, кэш, троттлинг.


def stem_tokens(tokens: List[str], stemmer: Stemmer) -> List[str]:
    """Применяет стеммер к списку токенов."""
    return [stemmer.stem(t) for t in tokens]


def stem_text(text: str, stemmer: Stemmer) -> List[str]:
    """Токенизирует и стеммит текст — удобный единый вход для пайплайнов."""
    return stem_tokens(tokenize(text), stemmer)


def get_stemmer(name: str = "identity") -> Stemmer:
    """
    Фабрика. name="identity" → baseline без стеммера;
    name="kazakh" → твой стеммер (после того как заполнишь KazakhStemmer).
    """
    if name == "identity":
        return IdentityStemmer()
    if name == "kazakh":
        # импорт здесь, чтобы baseline не тянул сетевой клиент
        from .stemmer_api import KazakhStemmerAPI

        return KazakhStemmerAPI()
    raise ValueError(f"Неизвестный стеммер: {name}")

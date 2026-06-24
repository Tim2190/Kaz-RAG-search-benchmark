"""
Слой предобработки входа (input-preprocessing): 4 линии текст-в-текст.

Идея
----
Починить ВХОД «сломанных» эмбеддеров без дообучения, меняя только текст перед
подачей в модель. Проверяем 4 линии и сравниваем 1–3 против baseline (0):

    0. baseline      — сырой текст (точка отсчёта)
    1. stem          — казахский стеммер (морфологическая нормализация)
    2. latin         — транслитерация кириллица→латиница (стандарт 2021)
    3. stem_latin    — стеммер, затем транслитерация

КРИТИЧНО (симметрия): одну и ту же transform применяем И к запросу, И к корпусу.
Иначе латиница-запрос против кириллица-корпуса = искусственный ложный провал.
Поэтому build_transform(line) возвращает ОДНУ функцию text→text, которую
run_preprocessing вешает на обе стороны.

Префиксы моделей (e5 "query: "/"passage: ", Qwen3 "Instruct: …") НЕ проходят
через этот слой — они английские инструкции и приклеиваются в DenseIndex уже
ПОСЛЕ предобработки казахского текста.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional

from .tokenize import _TOKEN_RE
from .translit import transliterate

#: канонические имена линий (используются в путях и таблицах результатов)
LINES = ("baseline", "stem", "latin", "stem_latin")


def stem_inline(text: str, stemmer) -> str:
    """
    Заменяет каждое слово-токен на его основу, СОХРАНЯЯ пунктуацию и структуру.

    В отличие от tokenize()+join (который теряет пунктуацию), здесь идёт
    re.sub по тому же токен-регэкспу: для dense-эмбеддингов важнее сохранить
    форму текста, а не превращать его в bag-of-stems. Регистр приводится к
    нижнему — стем-кэш построен на lowercase-токенах.
    """
    def _repl(m):
        return stemmer.stem(m.group(0).lower())
    return _TOKEN_RE.sub(_repl, text)


def collect_tokens(texts: Iterable[str]) -> List[str]:
    """Уникальные lowercase-токены по набору текстов — для прогрева стем-кэша."""
    seen = dict()  # сохраняем порядок, дедупликация
    for t in texts:
        for tok in _TOKEN_RE.findall(t.lower()):
            seen.setdefault(tok, None)
    return list(seen.keys())


def build_transform(line: str, stemmer=None) -> Callable[[str], str]:
    """
    Возвращает функцию text→text для указанной линии.

    line ∈ LINES. Для линий со стеммером ('stem', 'stem_latin') нужен
    прогретый stemmer (см. get_stemmer + warm). baseline → identity.
    """
    if line == "baseline":
        return lambda t: t
    if line == "latin":
        return transliterate
    if line == "stem":
        if stemmer is None:
            raise ValueError("линия 'stem' требует stemmer")
        return lambda t: stem_inline(t, stemmer)
    if line == "stem_latin":
        if stemmer is None:
            raise ValueError("линия 'stem_latin' требует stemmer")
        return lambda t: transliterate(stem_inline(t, stemmer))
    raise ValueError(f"неизвестная линия: {line!r}. Доступны: {LINES}")


def needs_stemmer(line: str) -> bool:
    """True, если линия использует стеммер (нужен прогрев кэша/сеть)."""
    return line in ("stem", "stem_latin")

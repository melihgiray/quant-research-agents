"""Small BM25 evidence index; no embedding model or remote download required."""

import json
import math
import re
from collections import Counter
from importlib.resources import files

from .evidence import EvidenceDocument, content_digest


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class EvidenceIndex:
    def __init__(self, documents: list[dict] | None = None):
        raw = (
            documents
            if documents is not None
            else json.loads(
                files("quant_research_agents").joinpath("knowledge/methodology.json").read_text()
            )
        )
        documents = [EvidenceDocument.model_validate(d).model_dump() for d in raw]
        if len({d["id"] for d in documents}) != len(documents):
            raise ValueError("evidence document IDs must be unique")
        self._documents = tuple({**d, "content_sha256": content_digest(d)} for d in documents)
        self.corpus_sha256 = content_digest(sorted(documents, key=lambda d: d["id"]))
        self.tokens = [Counter(tokenize(d["title"] + " " + d["text"])) for d in self._documents]
        self.lengths = [sum(t.values()) for t in self.tokens]
        self.average = sum(self.lengths) / max(len(self.lengths), 1)
        self.df = Counter(word for tokens in self.tokens for word in tokens)

    @property
    def documents(self) -> list[dict]:
        return [dict(d) for d in self._documents]

    def search(self, query: str, limit: int = 3) -> list[dict]:
        if not 1 <= limit <= 5 or not query.strip() or len(query) > 1000:
            raise ValueError("query must be 1–1000 characters and limit 1–5")
        scores = []
        n = len(self._documents)
        for i, tokens in enumerate(self.tokens):
            score = 0.0
            for word in set(tokenize(query)):
                tf = tokens[word]
                if not tf:
                    continue
                idf = math.log(1 + (n - self.df[word] + 0.5) / (self.df[word] + 0.5))
                norm = tf + 1.5 * (0.25 + 0.75 * self.lengths[i] / self.average)
                score += idf * tf * 2.5 / norm
            if score > 0:
                scores.append((score, i))
        scores.sort(key=lambda item: (-item[0], self._documents[item[1]]["id"]))
        return [{**self._documents[i], "score": round(score, 6)} for score, i in scores[:limit]]

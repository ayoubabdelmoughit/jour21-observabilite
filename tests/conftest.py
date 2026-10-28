"""
Config partagee pour les tests pytest du projet jour21.

- Fixe OLLAMA_API_KEY AVANT le premier `import agent_graph` (le module leve
  une RuntimeError au chargement si la cle est absente - voir agent_graph.py).
  Les tests ne font JAMAIS de vrai appel reseau a Ollama : _llm() est mocke
  dans chaque test, donc une fausse cle suffit ici.
- Ajoute la racine du projet (parent de tests/) a sys.path pour que
  `import agent_graph` fonctionne quel que soit le repertoire depuis lequel
  `pytest` est lance.
- Fournit une fausse collection MongoDB en memoire (FakeCollection), pour ne
  JAMAIS toucher la vraie base pendant les tests.
"""

import os
import sys

os.environ.setdefault("OLLAMA_API_KEY", "test-key-for-pytest-do-not-use-in-prod")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools
from types import SimpleNamespace

import pytest


class FakeCursor:
    """Imite juste assez l'API chainee de pymongo (find().sort().limit())
    pour que agent_graph.load_history() fonctionne sans vraie base."""

    def __init__(self, docs):
        self._docs = list(docs)

    def sort(self, key, direction=1):
        self._docs.sort(key=lambda d: d.get(key, 0), reverse=(direction == -1))
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    def __iter__(self):
        return iter(self._docs)


class FakeCollection:
    """Fausse collection MongoDB en memoire : insert_one + find basique,
    suffisant pour tester agent_graph.py sans dependance reseau/MongoDB."""

    def __init__(self):
        self.docs = []
        self._ids = itertools.count(1)

    def insert_one(self, doc):
        doc = dict(doc)
        doc["_id"] = f"fake-id-{next(self._ids)}"
        self.docs.append(doc)
        return SimpleNamespace(inserted_id=doc["_id"])

    def find(self, query=None):
        query = query or {}
        matched = [
            d for d in self.docs
            if all(d.get(k) == v for k, v in query.items())
        ]
        return FakeCursor(matched)


@pytest.fixture
def fake_collection():
    return FakeCollection()


@pytest.fixture
def fake_dlq_collection():
    return FakeCollection()


def make_fake_llm(content=None, side_effect=None):
    """Construit un faux objet "ChatOllama" : .invoke(messages) renvoie soit
    un objet avec .content = `content`, soit leve `side_effect` (pour
    simuler une panne/timeout cote LLM)."""

    def _invoke(messages):
        if side_effect is not None:
            raise side_effect
        return SimpleNamespace(content=content)

    return SimpleNamespace(invoke=_invoke)

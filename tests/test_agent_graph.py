"""
Tests automatises de l'agent de moderation (agent_graph.py) - Jour 22.

4 familles de tests demandees dans le cours (Tests Automatises & CI/CD
Agentique) :

1. Test de format      -> test_format_*
2. Test d'integration   -> test_integration_*
3. Test de policy       -> test_policy_*
4. Test de mauvaise entree (robustesse) -> test_robustness_*

Aucun test ne fait de vrai appel reseau (ni Ollama, ni MongoDB) : le LLM est
mocke via `_llm` (voir conftest.make_fake_llm) et MongoDB est remplace par
une fausse collection en memoire (conftest.FakeCollection). On peut donc
lancer `pytest` hors-ligne, sans cle API reelle ni cluster Mongo.
"""

from types import SimpleNamespace

import agent_graph
from conftest import make_fake_llm


# --------------------------------------------------------------- 1. FORMAT

def test_format_decision_line_is_parsed_correctly(monkeypatch, fake_collection):
    """Une reponse LLM bien formee doit produire un decision_log exploitable
    et une categorie `decision` correcte dans le document sauvegarde."""
    monkeypatch.setattr(
        agent_graph,
        "_llm",
        lambda temperature=0.1: make_fake_llm(
            content="DECISION: TOXIQUE | RAISON: Insulte directe envers l'utilisateur."
        ),
    )
    graph = agent_graph.build_graph(fake_collection)

    result = agent_graph.run_moderation(graph, "Tu es vraiment nul", session_id="s-format")

    assert result["decision_log"].startswith("DECISION: TOXIQUE")
    assert "duration_ms" in result
    assert isinstance(result["duration_ms"], float)
    saved = fake_collection.docs[-1]
    assert saved["decision"] == "TOXIQUE"
    assert saved["text"] == "Tu es vraiment nul"
    assert "inserted_id" in result


def test_format_saved_document_has_required_fields(monkeypatch, fake_collection):
    """Le document ecrit dans MongoDB doit toujours contenir les champs
    attendus par /metrics (dashboard_backend_v2.py)."""
    monkeypatch.setattr(
        agent_graph,
        "_llm",
        lambda temperature=0.1: make_fake_llm(content="DECISION: OK | RAISON: Message neutre."),
    )
    graph = agent_graph.build_graph(fake_collection)
    agent_graph.run_moderation(graph, "Bonjour tout le monde", session_id="s-format-2")

    saved = fake_collection.docs[-1]
    champs_requis = {
        "agent_id", "decision_log", "decision", "text", "session_id",
        "duration_ms", "cost_usd", "timestamp",
    }
    assert champs_requis.issubset(saved.keys())
    assert saved["cost_usd"] == 0.0


# ---------------------------------------------------------- 2. INTEGRATION

def test_integration_history_flows_between_nodes(monkeypatch, fake_collection):
    """Verifie l'enchainement load_history -> classify -> save_log : le 2e
    message de la MEME session doit recevoir l'historique du 1er dans son
    prompt (c'est le but du noeud load_history)."""
    captured_prompts = []

    def _invoke(messages):
        # messages[1] est le HumanMessage contenant le prompt complet
        captured_prompts.append(messages[1].content)
        return SimpleNamespace(content="DECISION: SPAM | RAISON: Publicite repetee.")

    monkeypatch.setattr(agent_graph, "_llm", lambda temperature=0.1: SimpleNamespace(invoke=_invoke))
    graph = agent_graph.build_graph(fake_collection)

    agent_graph.run_moderation(graph, "Achetez maintenant !", session_id="s-integ")
    agent_graph.run_moderation(graph, "Achetez maintenant !", session_id="s-integ")

    assert len(captured_prompts) == 2
    assert "aucun message precedent" in captured_prompts[0]
    # Le 2e appel doit contenir l'historique du 1er message (meme session)
    assert "Achetez maintenant !" in captured_prompts[1]
    assert "-> SPAM" in captured_prompts[1]
    assert len(fake_collection.docs) == 2


def test_integration_sessions_are_isolated(monkeypatch, fake_collection):
    """Deux sessions differentes ne doivent PAS partager leur historique."""
    monkeypatch.setattr(
        agent_graph,
        "_llm",
        lambda temperature=0.1: make_fake_llm(content="DECISION: OK | RAISON: ok."),
    )
    graph = agent_graph.build_graph(fake_collection)

    agent_graph.run_moderation(graph, "message session A", session_id="session-A")
    result_b = agent_graph.run_moderation(graph, "message session B", session_id="session-B")

    # Le 2e message est dans une AUTRE session : load_history doit repartir
    # de zero, donc le document sauve pour session-B ne doit referencer que
    # session-B.
    assert result_b["inserted_id"]
    docs_b = [d for d in fake_collection.docs if d["session_id"] == "session-B"]
    assert len(docs_b) == 1


# --------------------------------------------------------------- 3. POLICY

def test_policy_decision_is_always_a_known_category(monkeypatch, fake_collection):
    """Quelle que soit la reponse du LLM (bien formee, vide, ou n'importe
    quoi), la `decision` finale doit TOUJOURS appartenir a l'ensemble connu
    {OK, TOXIQUE, SPAM, INCONNU} - jamais une valeur libre non prevue par le
    dashboard."""
    cas = [
        ("DECISION: OK | RAISON: ras.", "OK"),
        ("DECISION: SPAM | RAISON: pub.", "SPAM"),
        ("DECISION: TOXIQUE | RAISON: insulte.", "TOXIQUE"),
        ("reponse totalement hors format, sans mot-cle connu", "INCONNU"),
        ("", "INCONNU"),
    ]
    categories_valides = {"OK", "TOXIQUE", "SPAM", "INCONNU"}

    for llm_output, expected in cas:
        monkeypatch.setattr(
            agent_graph, "_llm", lambda temperature=0.1, c=llm_output: make_fake_llm(content=c)
        )
        graph = agent_graph.build_graph(fake_collection)
        agent_graph.run_moderation(graph, "texte de test", session_id="s-policy")
        decision = fake_collection.docs[-1]["decision"]
        assert decision in categories_valides
        assert decision == expected


def test_policy_cost_is_always_zero(monkeypatch, fake_collection):
    """Regle metier : le cout doit toujours etre 0.0 (Ollama self-heberge),
    jamais une estimation inventee."""
    monkeypatch.setattr(
        agent_graph,
        "_llm",
        lambda temperature=0.1: make_fake_llm(content="DECISION: OK | RAISON: ok."),
    )
    graph = agent_graph.build_graph(fake_collection)
    agent_graph.run_moderation(graph, "n'importe quel texte", session_id="s-policy-cost")
    assert fake_collection.docs[-1]["cost_usd"] == 0.0


# ----------------------------------------------- 4. ROBUSTESSE / MAUVAISE ENTREE

def test_robustness_empty_text_does_not_crash(fake_collection):
    """Un texte vide (ou uniquement des espaces) ne doit jamais faire
    planter le graphe ni partir en appel LLM inutile."""
    graph = agent_graph.build_graph(fake_collection)
    result = agent_graph.run_moderation(graph, "   ", session_id="s-robust-empty")

    assert result["error"] == "empty_text"
    assert fake_collection.docs[-1]["decision"] == "INCONNU"


def test_robustness_llm_failure_is_caught_and_sent_to_dlq(monkeypatch, fake_collection, fake_dlq_collection):
    """Simule une panne du LLM (timeout/erreur 500 equivalent). Le graphe ne
    doit pas lever d'exception, et la requete en echec doit atterrir dans la
    DLQ (Dead Letter Queue) pour pouvoir etre rejouee plus tard."""
    monkeypatch.setattr(
        agent_graph,
        "_llm",
        lambda temperature=0.1: make_fake_llm(side_effect=TimeoutError("simulated upstream timeout")),
    )
    graph = agent_graph.build_graph(fake_collection, fake_dlq_collection)

    result = agent_graph.run_moderation(graph, "message tout a fait normal", session_id="s-robust-panne")

    assert result.get("error")
    assert "simulated upstream timeout" in result["decision_log"]
    # Le log normal existe quand meme (traçabilite), categorise INCONNU.
    assert fake_collection.docs[-1]["decision"] == "INCONNU"
    # Et la DLQ a bien recu une copie de la requete en echec.
    assert len(fake_dlq_collection.docs) == 1
    assert fake_dlq_collection.docs[0]["text"] == "message tout a fait normal"
    assert "simulated upstream timeout" in fake_dlq_collection.docs[0]["error"]
    assert fake_dlq_collection.docs[0]["retried"] is False


def test_robustness_without_dlq_collection_still_works(monkeypatch, fake_collection):
    """Si aucune DLQ n'est fournie (dlq_collection=None, comportement par
    defaut), une panne LLM ne doit toujours pas faire planter le graphe."""
    monkeypatch.setattr(
        agent_graph,
        "_llm",
        lambda temperature=0.1: make_fake_llm(side_effect=RuntimeError("boom")),
    )
    graph = agent_graph.build_graph(fake_collection)  # pas de DLQ
    result = agent_graph.run_moderation(graph, "texte", session_id="s-robust-no-dlq")
    assert result.get("error")
    assert fake_collection.docs[-1]["decision"] == "INCONNU"


def test_robustness_very_long_text_is_handled(monkeypatch, fake_collection):
    """Un texte anormalement long ne doit pas faire planter le graphe."""
    monkeypatch.setattr(
        agent_graph,
        "_llm",
        lambda temperature=0.1: make_fake_llm(content="DECISION: SPAM | RAISON: texte suspect tres long."),
    )
    graph = agent_graph.build_graph(fake_collection)
    texte_long = "a" * 20000
    result = agent_graph.run_moderation(graph, texte_long, session_id="s-robust-long")
    assert result["decision_log"].startswith("DECISION: SPAM")

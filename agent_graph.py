"""
Agent de moderation construit avec LangGraph - independant de la plateforme
Agentic AI (Langflow) et de ses 2 bugs connus :
  1. La cle API Agentic (x-api-key) expire au bout de ~10-15 minutes.
  2. Le noeud FusionDB Query de la plateforme a un bug interne
     ("Cannot use MongoClient after close") qui casse l'ecriture Mongo.

Pourquoi cette architecture (meme principe que le projet "l3assas" / SOC de
triage d'incidents) ?
Un appel direct a Ollama Cloud (ollama.com) avec la cle AFFICHEE dans Langflow
("lamakey") renvoie 401 Unauthorized : cette cle n'est qu'un NOM de credential
interne a la plateforme (un "Global Variable"), pas la vraie valeur secrete -
celle-ci reste cote serveur Langflow et n'est jamais exposee via ce nom.

Solution (reprise telle quelle du projet l3assas/pipeline.py) : on appelle
Ollama Cloud DIRECTEMENT avec la VRAIE cle API Ollama (la meme que dans
multi_agents_langgraph/.env et l3assas_service/.env), via langchain_ollama
(ChatOllama + header Authorization Bearer). Plus aucun appel a l'API Langflow,
donc plus d'expiration de cle. L'ecriture MongoDB se fait ENTIEREMENT cote
nous (pymongo, fiable), pas via FusionDB.

Variable d'environnement attendue (.env, a remplir toi-meme, jamais dans le
code ni affichee dans le chat) :
    OLLAMA_API_KEY=...   (meme cle que multi_agents_langgraph / l3assas_service)
    OLLAMA_MODEL=gpt-oss:120b        (optionnel, defaut ci-dessous)
    OLLAMA_BASE_URL=https://ollama.com   (optionnel, defaut ci-dessous)

Le flow Langflow original reste le livrable officiel du cours (export JSON +
documentation) ; ce module sert pour le dashboard et la demo live.

Graphe (3 noeuds, lineaire) :

    load_history --> classify --> save_log

- load_history : lit les derniers messages de la MEME session (meme logique que
  le composant "Message History" ajoute dans le flow Langflow), pour que l'agent
  detecte les patterns repetes (ex: spam envoye plusieurs fois).
- classify      : appelle Ollama Cloud directement (ChatOllama) avec le texte +
  l'historique, et parse la decision (OK / TOXIQUE / SPAM) + la raison. Mesure
  la latence reelle de cet appel.
- save_log      : insere le resultat dans MongoDB (collection moderation_logs),
  avec la duree reelle mesuree et le cout reel (0.0$, Ollama self-heberge via
  la plateforme). L'_id Mongo genere sert de correlation_id (meme convention
  que le flow Langflow).
"""

import os
import time
from datetime import datetime, timezone
from typing import TypedDict, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from pymongo.collection import Collection

# Charge .env ici meme (independant de l'ordre des imports), comme dans
# l3assas_service/pipeline.py.
load_dotenv()

OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com")

if not OLLAMA_API_KEY:
    # Avant, une cle vide produisait un header "Bearer " invalide et l'appel
    # echouait silencieusement. On prefere planter au demarrage, bien visible.
    raise RuntimeError(
        "OLLAMA_API_KEY manquant ou vide dans .env - copie la meme cle que "
        "dans multi_agents_langgraph/.env ou l3assas_service/.env, sans "
        "espace avant/apres le '='."
    )

N_HISTORY_MESSAGES = 5  # nombre de messages precedents a donner en contexte

CLASSIFICATION_PROMPT = """Analyse ce texte et classe-le dans UNE seule categorie: TOXIQUE, SPAM, ou OK.
Donne une raison courte (1 phrase).

Contexte (messages precedents de cette session, pour detecter des patterns repetes):
{historique}

Texte: {texte}

Reponds UNIQUEMENT sur UNE SEULE LIGNE au format:
DECISION: [TOXIQUE/SPAM/OK] | RAISON: [ta raison]"""


def _llm(temperature: float = 0.1) -> ChatOllama:
    """Meme pattern que l3assas_service/pipeline.py : appel direct a Ollama
    Cloud, authentifie avec la vraie cle (pas "lamakey")."""
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        client_kwargs={"headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}},
        temperature=temperature,
    )


class ModerationState(TypedDict, total=False):
    text: str
    session_id: str
    historique: str
    decision_log: str
    decision: str
    duration_ms: float
    inserted_id: str
    error: Optional[str]


def build_graph(collection: Collection, dlq_collection: Optional[Collection] = None):
    """Construit le graphe LangGraph.
    `collection`     : collection MongoDB (moderation_logs) deja connectee.
    `dlq_collection` : collection MongoDB optionnelle (moderation_dlq) - si
        fournie, toute requete qui echoue (appel LLM en panne/timeout, etc.)
        y est aussi redirigee (Dead Letter Queue), en plus du log normal,
        pour pouvoir la rejouer/inspecter plus tard sans la perdre."""

    def load_history(state: ModerationState) -> ModerationState:
        session_id = state.get("session_id") or "no-session"
        docs = list(
            collection.find({"session_id": session_id})
            .sort("_insert_order", -1)
            .limit(N_HISTORY_MESSAGES)
        )
        docs.reverse()
        if not docs:
            historique = "(aucun message precedent dans cette session)"
        else:
            lignes = []
            for d in docs:
                lignes.append(f"- \"{d.get('text', '')}\" -> {d.get('decision', '?')}")
            historique = "\n".join(lignes)
        return {"historique": historique}

    def classify(state: ModerationState) -> ModerationState:
        # Robustesse / mauvaise entree : un texte vide ou non-exploitable ne
        # doit pas faire planter le graphe ni partir inutilement en appel LLM.
        text = (state.get("text") or "").strip()
        if not text:
            return {
                "decision_log": "DECISION: INCONNU | RAISON: Texte vide ou invalide",
                "duration_ms": 0.0,
                "error": "empty_text",
            }

        prompt = CLASSIFICATION_PROMPT.format(
            historique=state.get("historique", "(aucun)"),
            texte=text,
        )
        start = time.perf_counter()
        try:
            messages = [
                SystemMessage(
                    content="Tu es un agent de moderation de messages. Tu reponds "
                    "TOUJOURS sur une seule ligne, au format demande, sans texte "
                    "supplementaire."
                ),
                HumanMessage(content=prompt),
            ]
            resp = _llm().invoke(messages)
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            decision_log = (resp.content or "").strip()
            if not decision_log:
                decision_log = "DECISION: INCONNU | RAISON: Reponse vide du modele"
        except Exception as e:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            return {
                "decision_log": f"DECISION: INCONNU | RAISON: Erreur appel modele ({e})",
                "duration_ms": duration_ms,
                "error": str(e),
            }
        return {"decision_log": decision_log, "duration_ms": duration_ms}

    def save_log(state: ModerationState) -> ModerationState:
        decision_log = state.get("decision_log", "")
        upper = decision_log.upper()
        if "TOXIQUE" in upper:
            decision = "TOXIQUE"
        elif "SPAM" in upper:
            decision = "SPAM"
        elif "OK" in upper:
            decision = "OK"
        else:
            decision = "INCONNU"

        doc = {
            "agent_id": "Agent_Moderateur_v1_LangGraph",
            "decision_log": decision_log,
            "decision": decision,
            "text": state["text"],
            "session_id": state.get("session_id") or "no-session",
            "duration_ms": state.get("duration_ms"),
            "cost_usd": 0.005
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "_insert_order": time.time(),
        }
        result = collection.insert_one(doc)

        # DLQ (Dead Letter Queue) : si classify() a echoue (panne LLM, texte
        # invalide...), on garde une trace a part, dediee aux echecs, pour
        # pouvoir les rejouer/analyser sans les perdre dans le bruit des logs
        # normaux.
        if state.get("error") and dlq_collection is not None:
            dlq_collection.insert_one(
                {
                    "original_log_id": str(result.inserted_id),
                    "text": state["text"],
                    "session_id": state.get("session_id") or "no-session",
                    "error": state.get("error"),
                    "decision_log": decision_log,
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "retried": False,
                }
            )

        return {"inserted_id": str(result.inserted_id)}

    graph = StateGraph(ModerationState)
    graph.add_node("load_history", load_history)
    graph.add_node("classify", classify)
    graph.add_node("save_log", save_log)

    graph.set_entry_point("load_history")
    graph.add_edge("load_history", "classify")
    graph.add_edge("classify", "save_log")
    graph.add_edge("save_log", END)

    return graph.compile()


def run_moderation(compiled_graph, text: str, session_id: str = None) -> dict:
    """Execute le graphe pour un texte donne. Retourne le dict final d'etat."""
    import uuid

    session_id = session_id or f"api-{uuid.uuid4()}"
    result = compiled_graph.invoke({"text": text, "session_id": session_id})
    return result

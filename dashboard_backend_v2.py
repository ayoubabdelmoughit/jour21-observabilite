"""
Backend Jour21 - Observabilite des agents (v2 : LangGraph, independant de la
plateforme Agentic AI)

Pourquoi cette version ?
La version precedente appelait l'agent via l'API de la plateforme Agentic AI
(Langflow), ce qui a pose 2 problemes cote plateforme (hors de notre controle) :
  1. Les cles API Agentic generees expirent au bout de ~10-15 minutes.
  2. Le composant FusionDB Query de la plateforme a un bug interne
     ("Cannot use MongoClient after close") qui casse l'ecriture Mongo.

Cette version remplace l'appel a la plateforme par un agent LangGraph local
(agent_graph.py) qui appelle directement Ollama Cloud avec la VRAIE cle API
Ollama (meme principe et meme cle que le projet "l3assas" / SOC de triage
d'incidents : voir l3assas_service/pipeline.py) et ecrit directement dans
MongoDB avec pymongo. Resultat : plus aucune dependance a la plateforme pour
le test live, donc plus d'expiration ni de bug MongoClient cote plateforme.

Le flow Langflow original reste le livrable officiel du cours (export JSON +
documentation) ; ce backend sert pour le dashboard et la demo live.

INSTALLATION (une seule fois):
    pip install -r requirements.txt --break-system-packages

CONFIGURATION (une seule fois):
    Copier .env.example en .env, et coller la vraie valeur de OLLAMA_API_KEY
    (meme cle que multi_agents_langgraph/.env ou l3assas_service/.env).
    Ne jamais coller cette cle dans le chat.

LANCEMENT:
    python3 dashboard_backend_v2.py

Puis, dans un AUTRE terminal, exposer le port avec ngrok:
    ngrok http 8010

Colle l'URL ngrok + /metrics dans le dashboard (onglet Connexion API).
Le panneau "Tester l'agent" (chat) appelle /moderate pour chaque message.
"""

import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pymongo import MongoClient

from agent_graph import build_graph, run_moderation

# Dossier ou se trouve ce script (= ou se trouve dashboard.html a cote).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ====== Configuration (deja remplie) ======
MONGO_URI = "mongodb+srv://ayoubabdelmoughit75_db_user:gxl2vJVaC2was3fC@ayoub.eknyomo.mongodb.net/?appName=ayoub"
DB_NAME = "jour21_observabilite"
COLLECTION_NAME = "moderation_logs"

LATENCE_SEUIL_MS = 3000
# ============================================

app = FastAPI(title="Jour21 Observabilite Backend v2 (LangGraph)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]
dlq_collection = client[DB_NAME]["moderation_dlq"]
moderation_graph = build_graph(collection, dlq_collection)


@app.post("/moderate")
def moderate(payload: dict):
    """Envoie un texte a l'agent LangGraph, mesure la latence reelle, logge dans MongoDB."""
    text = (payload or {}).get("text", "").strip()
    session_id = (payload or {}).get("session_id") or None
    if not text:
        return {"error": "Le champ 'text' est requis."}

    result = run_moderation(moderation_graph, text, session_id)

    if result.get("error") and not result.get("inserted_id"):
        return {"error": result["error"], "duration_ms": result.get("duration_ms")}

    return {
        "id": result.get("inserted_id"),
        "decision_log": result.get("decision_log"),
        "duration_ms": result.get("duration_ms"),
        "cost_usd": 0.0,
        "session_id": session_id,
    }


@app.get("/metrics")
def get_metrics():
    docs = list(collection.find({}).sort("_insert_order", 1))
    total = len(docs)

    counts = {"OK": 0, "TOXIQUE": 0, "SPAM": 0, "INCONNU": 0}
    logs = []
    durations = []
    total_cost = 0.0

    for doc in docs:
        decision_log = doc.get("decision_log", "")
        category = doc.get("decision")
        if not category:
            upper = decision_log.upper()
            if "TOXIQUE" in upper:
                category = "TOXIQUE"
            elif "SPAM" in upper:
                category = "SPAM"
            elif "DECISION: OK" in upper or "DECISION:OK" in upper:
                category = "OK"
            else:
                category = "INCONNU"
        counts[category] = counts.get(category, 0) + 1

        duration = doc.get("duration_ms")
        if isinstance(duration, (int, float)):
            durations.append(duration)

        cost = doc.get("cost_usd")
        if isinstance(cost, (int, float)):
            total_cost += cost

        logs.append({
            "id": str(doc.get("_id", "")),
            "agent_id": doc.get("agent_id", "N/A"),
            "decision": category,
            "raw": decision_log,
            "duration_ms": duration,
            "timestamp": doc.get("timestamp", ""),
            "session_id": doc.get("session_id", ""),
        })

    valid = counts["OK"] + counts["TOXIQUE"] + counts["SPAM"]
    taux_succes = round((valid / total) * 100, 1) if total else 0
    tox_pct = round((counts["TOXIQUE"] / total) * 100, 1) if total else 0
    ok_pct = round((counts["OK"] / total) * 100, 1) if total else 0
    spam_pct = round((counts["SPAM"] / total) * 100, 1) if total else 0

    avg_latency = round(sum(durations) / len(durations), 1) if durations else None
    nb_mesures_latence = len(durations)

    alerts = {
        "taux_echec": {
            "condition": "Taux de succes < 80%",
            "declenchee": taux_succes < 80,
        },
        "volume_toxique": {
            "condition": "Part TOXIQUE > 30%",
            "declenchee": tox_pct > 30,
        },
        "latence_elevee": {
            "condition": f"Latence moyenne > {LATENCE_SEUIL_MS}ms",
            "declenchee": (avg_latency is not None and avg_latency > LATENCE_SEUIL_MS),
        },
    }

    return {
        "total": total,
        "taux_succes": taux_succes,
        "ok": {"count": counts["OK"], "pct": ok_pct},
        "toxique": {"count": counts["TOXIQUE"], "pct": tox_pct},
        "spam": {"count": counts["SPAM"], "pct": spam_pct},
        "avg_latency_ms": avg_latency,
        "nb_mesures_latence": nb_mesures_latence,
        "total_cost_usd": round(total_cost, 4),
        "cost_note": "Ollama self-hebergee (cle gratuite) : cout reel mesure, pas une estimation.",
        "alerts": alerts,
        "logs": logs,
    }


@app.get("/dlq")
def get_dlq():
    """Dead Letter Queue : liste les requetes qui ont echoue (panne LLM,
    texte invalide...) et n'ont donc pas ete classifiees normalement. Permet
    de les inspecter/rejouer sans les avoir perdues."""
    docs = list(dlq_collection.find({}).sort("timestamp", -1))
    for d in docs:
        d["id"] = str(d.pop("_id", ""))
    return {"total": len(docs), "items": docs}


@app.get("/")
def root():
    """Sert dashboard.html directement sur la racine du backend : on ouvre
    une seule URL (locale ou ngrok) et la page + l'API sont sur la meme
    origine, donc plus besoin de copier-coller l'URL ngrok a la main dans
    le champ "Connexion API" (meme principe que l3assas : frontend servi
    par le meme serveur que l'API)."""
    dashboard_path = os.path.join(BASE_DIR, "dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    return {"status": "ok", "message": "Backend Jour21 v2 (LangGraph) actif. Essayez /metrics ou POST /moderate"}


if __name__ == "__main__":
    print("Demarrage sur http://localhost:8010 ...")
    print("Endpoint metrics  : http://localhost:8010/metrics")
    print("Endpoint moderate : POST http://localhost:8010/moderate  body: {\"text\": \"...\", \"session_id\": \"...\"}")
    uvicorn.run(app, host="0.0.0.0", port=8010)

from typing import Dict, Any, Optional, List
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---- import your existing code ----
from src.config import TEST_PATH
from src.agent1 import agent1_node
from src.agent2_runtime import load_agent2_artifacts, run_agent2, compute_shap_for_instance
from src.agent3_explainer import generate_explanation, generate_followup_explanation

app = FastAPI(title="Dispute Wizard API")

# CORS for local dev front-end (adjust for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts once
load_agent2_artifacts()

# Very simple in-memory session store (OK for demo/dev).
# For production: Redis or DB keyed by user/session token.
SESSIONS: Dict[str, Dict[str, Any]] = {}


def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "stage": 0,
            "raw_input": {},
            "enriched_features": {},
            "feature_explanations": {},
            "agent2_predictions": {},
            "agent2_shap_top_features": [],
            "nl_explanation": "",
            "conversation_history": "",
            "human_feedback": {
                "approved_enrichment": None,
                "approved_explanation": None,
            },
        }
    return SESSIONS[session_id]


# ---------- Schemas ----------

class SessionOnlyReq(BaseModel):
    session_id: str

class Agent2RunReq(BaseModel):
    session_id: str
    enriched_features_override: Optional[Dict[str, Any]] = None

class Agent2ShapReq(BaseModel):
    session_id: str
    enriched_features_override: Optional[Dict[str, Any]] = None


class SessionReq(BaseModel):
    session_id: str

class SelectReq(BaseModel):
    session_id: str
    dispute_id: Optional[Any] = None
    row_index: Optional[int] = None

class OverrideReq(BaseModel):
    session_id: str
    patch: Dict[str, Any]  # fields to override, e.g. {"mcc": 1234, "txn_amount": 99.5}

class AskReq(BaseModel):
    session_id: str
    question: str


# ---------- Helpers ----------
def load_test_df() -> pd.DataFrame:
    if not os.path.exists(TEST_PATH):
        raise HTTPException(status_code=500, detail=f"Missing file: {TEST_PATH}")
    return pd.read_csv(TEST_PATH)


@app.post("/session/reset")
def session_reset(req: SessionReq):
    SESSIONS.pop(req.session_id, None)
    return {"ok": True}

@app.get("/disputes")
def list_disputes(limit: int = 500):
    df = load_test_df()
    cols = df.columns.tolist()
    id_col = "dispute_id" if "dispute_id" in cols else None
    items = []
    if id_col:
        for v in df[id_col].head(limit).tolist():
            items.append({"dispute_id": v})
    else:
        for i in range(min(limit, len(df))):
            items.append({"row_index": i})
    return {"has_dispute_id": bool(id_col), "items": items, "columns": cols}

@app.post("/select")
def select_dispute(req: SelectReq):
    s = get_session(req.session_id)
    df = load_test_df()
    id_col = "dispute_id" if "dispute_id" in df.columns else None

    if id_col and req.dispute_id is not None:
        row = df[df[id_col] == req.dispute_id]
        if row.empty:
            raise HTTPException(404, "dispute_id not found")
        raw = row.iloc[0].to_dict()
    else:
        idx = req.row_index if req.row_index is not None else 0
        if idx < 0 or idx >= len(df):
            raise HTTPException(400, "row_index out of range")
        raw = df.iloc[int(idx)].to_dict()

    # reset downstream
    s.update({
        "stage": 0,
        "raw_input": raw,
        "enriched_features": {},
        "feature_explanations": {},
        "agent2_predictions": {},
        "agent2_shap_top_features": [],
        "nl_explanation": "",
        "conversation_history": "",
        "human_feedback": {"approved_enrichment": None, "approved_explanation": None},
    })
    return {"raw_input": s["raw_input"]}

@app.post("/agent1/run")
def agent1_run(req: SessionReq):
    s = get_session(req.session_id)
    if not s["raw_input"]:
        raise HTTPException(400, "No raw_input selected")

    state = {"raw_input": s["raw_input"]}
    state = agent1_node(state)
    s["enriched_features"] = state["enriched_features"]
    s["feature_explanations"] = state["feature_explanations"]
    s["human_feedback"]["approved_enrichment"] = None
    return {
        "enriched_features": s["enriched_features"],
        "feature_explanations": s["feature_explanations"],
    }

@app.post("/agent1/override")
def agent1_override(req: OverrideReq):
    s = get_session(req.session_id)
    if not s["raw_input"]:
        raise HTTPException(400, "No raw_input selected")

    # apply patch
    s["raw_input"] = {**s["raw_input"], **req.patch}

    # reset downstream + rerun agent1 like Streamlit
    s["enriched_features"] = {}
    s["feature_explanations"] = {}
    s["agent2_predictions"] = {}
    s["agent2_shap_top_features"] = []
    s["nl_explanation"] = ""
    s["conversation_history"] = ""
    s["human_feedback"] = {"approved_enrichment": None, "approved_explanation": None}

    state = {"raw_input": s["raw_input"]}
    state = agent1_node(state)
    s["enriched_features"] = state["enriched_features"]
    s["feature_explanations"] = state["feature_explanations"]
    return {
        "raw_input": s["raw_input"],
        "enriched_features": s["enriched_features"],
        "feature_explanations": s["feature_explanations"],
    }

@app.post("/agent2/run")
def agent2_run(req: Agent2RunReq):
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    # IMPORTANT: prefer override if provided
    feats = req.enriched_features_override or session.get("enriched_features")
    if not feats:
        raise HTTPException(status_code=400, detail="Missing enriched features. Run agent1 first.")

    preds, shap_top = run_agent2(feats)

    # optional: persist predictions/shap in session
    session["agent2_predictions"] = preds
    session["agent2_shap_top_features"] = shap_top

    return {"predictions": preds, "shap_top": shap_top}



@app.post("/agent2/shap")
def agent2_shap(req: Agent2ShapReq):
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    feats = req.enriched_features_override or session.get("enriched_features")
    if not feats:
        raise HTTPException(status_code=400, detail="Missing enriched features. Run agent1 first.")

    shap_vec, feat_cols = compute_shap_for_instance(feats)

    # Build "top" list like your UI expects
    import numpy as np
    abs_vals = np.abs(shap_vec)
    top_idx = np.argsort(-abs_vals)[:10]

    top = []
    for i in top_idx:
        top.append({
            "feature": feat_cols[i],
            "value": float(shap_vec[i]),
        })

    return {"top": top}



@app.post("/agent3/explain")
def agent3_explain(req: SessionReq):
    s = get_session(req.session_id)
    if not s["agent2_predictions"]:
        raise HTTPException(400, "Missing agent2_predictions")

    state = {
        "raw_input": s["raw_input"],
        "enriched_features": s["enriched_features"],
        "agent2_predictions": s["agent2_predictions"],
        "agent2_shap_top_features": s["agent2_shap_top_features"],
    }
    try:
        expl = generate_explanation(state)
    except Exception as e:
        expl = f"Unable to generate explanation at the moment due to service constraints. Please try again later. (Error: {type(e).__name__})"
    
    s["nl_explanation"] = expl
    s["conversation_history"] = expl
    s["human_feedback"]["approved_explanation"] = None
    return {"explanation": expl}

@app.post("/agent3/ask")
def agent3_ask(req: AskReq):
    s = get_session(req.session_id)
    if not s["agent2_predictions"]:
        raise HTTPException(400, "Missing agent2_predictions")
    if not s["conversation_history"]:
        raise HTTPException(400, "Generate explanation first")

    state = {
        "raw_input": s["raw_input"],
        "enriched_features": s["enriched_features"],
        "agent2_predictions": s["agent2_predictions"],
        "agent2_shap_top_features": s["agent2_shap_top_features"],
    }
    ans = generate_followup_explanation(
        state=state,
        previous_explanation=s["conversation_history"],
        user_question=req.question,
    )
    s["conversation_history"] += f"\n\n[Q] {req.question}\n[A] {ans}"
    s["nl_explanation"] += f"\n\nFollow-up:\n{ans}"
    return {"answer": ans, "conversation_history": s["conversation_history"], "explanation": s["nl_explanation"]}


# @app.exception_handler(Exception)
# async def global_exception_handler(request, exc):
#     # Keep safe, don’t leak secrets in prod
#     return JSONResponse(
#         status_code=500,
#         content={"detail": "Internal server error", "error": str(exc)}
#     )

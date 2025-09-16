import os
import re
import json
import time
from uuid import uuid4
from typing import Dict, List, Any

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import (
    ConversationSummaryMemory,
    ConversationBufferMemory,
    CombinedMemory,
)
from langchain.agents import Tool, AgentType, initialize_agent

# ================= ENV & CONFIG =================
load_dotenv()

DATA_FILE = os.getenv("DATA_FILE", "data/Inventory_augmented.xlsx")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini-2024-07-18")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
ROW_CHUNK_SIZE = int(os.getenv("ROW_CHUNK_SIZE", "15"))

# Thresholds for inlining vs caching large tables (to avoid hitting LLM context)
TABLE_INLINE_ROW_THRESHOLD = int(os.getenv("TABLE_INLINE_ROW_THRESHOLD", "80"))
TABLE_INLINE_JSON_CHARS = int(os.getenv("TABLE_INLINE_JSON_CHARS", "4000"))

# Pandas display (avoid truncation server-side; UI renders fully)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

app = Flask(__name__)

# ================= GLOBAL STATE =================
df: pd.DataFrame = None
vectorstore: Any = None

# Per-session caches
SESSION_MEMORIES: Dict[str, CombinedMemory] = {}
SESSION_RAG_CHAINS: Dict[str, ConversationalRetrievalChain] = {}
SESSION_AGENTS: Dict[str, Any] = {}
SESSION_LAST_TABLE: Dict[str, Dict[str, Any]] = {}

# Cache for large tables to avoid pushing huge payloads through LLM messages
TABLE_CACHE: Dict[str, Dict[str, Any]] = {}

# ================= LLM =================
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=LLM_TEMPERATURE,
    api_key=OPENAI_API_KEY,
    max_tokens=LLM_MAX_TOKENS,
)

# ================= PROMPTS =================
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a data documentation assistant. "
        "Use ONLY the provided context (schema descriptions, sample values, "
        "metadata, comments). "
        "Do NOT perform calculations, counts, or aggregations. "
        "Do NOT use external knowledge or the internet. "
        "Do NOT invent new columns or values. "
        "If the answer is not explicitly in the context, say "
        "'I don’t know based on the data.'\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer strictly based on context:"
    ),
)

NL_TO_PANDAS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You translate natural language questions into executable Pandas code.\n"
            "Rules:\n"
            "- Use ONLY the provided dataframe `df`.\n"
            "- Do NOT read/write files, import modules, or access network/OS.\n"
            "- Use case-insensitive matching for text filters: "
            "  df[col].astype(str).str.contains(<text>, case=False, na=False)\n"
            "- Normalize boolean-like text: 'true/yes/y/1' => True, "
            "'false/no/n/0' => False.\n"
            "- When counting, return pd.DataFrame({{'Count': [count_value]}}).\n"
            "- If the logical result is a scalar/list/series, convert it to a "
            "DataFrame.\n"
            "- If a previous table is available as `last_df`, and the user "
            "question clearly refers to the previous result or mentions a value/"
            "entity that likely came from that result, operate on `last_df` "
            "instead of `df`.\n"
            "- Final line must assign the result to a variable named result_df.\n"
            "- Never print; no plots; only compute result_df.\n",
        ),
        (
            "user",
            "Chat context (may be empty):\n{chat_context}\n\n"
            "User question:\n{question}\n\n"
            "Dataframe columns:\n{columns}\n\n"
            "Write only the Python code block (no explanation). "
            "Ensure the last non-empty line is: result_df = <DataFrame>",
        ),
    ]
)

REFINE_PROMPT = PromptTemplate(
    input_variables=["user_query"],
    template=(
        "You are a helpful assistant that improves vague user questions about a "
        "dataset.\n\n"
        'User query: "{user_query}"\n\n'
        "Your task: Suggest 3 clearer, more specific NATURAL LANGUAGE questions a "
        "business user could ask.\n\n"
        " Rules:\n"
        "- Use only plain English.\n"
        "- Do NOT generate SQL, code, or technical syntax.\n"
        "- Keep them one sentence each, short and clear.\n"
        "- Stay close in meaning to the user query.\n"
        "- Output ONLY a JSON array of 3 strings, no markdown, no explanation.\n\n"
        'Example: ["Which files include engagement data?", '
        '"How many records belong to brand Diab_1?", '
        '"List all datasets mentioning Diab_1 with engagement metrics"]'
    ),
)

# ================= FAST-PATHS (cheap shortcuts) =================
FAST_PATTERNS = [
    (re.compile(r"^(how\s+many\s+rows|row\s+count)\b", re.I), lambda _q: str(len(df))),
    (
        re.compile(r"^(how\s+many\s+columns|column\s+count)\b", re.I),
        lambda _q: str(len(df.columns)),
    ),
    (
        re.compile(r"^(list|show)\s+columns\b", re.I),
        lambda _q: json.dumps(list(map(str, df.columns.tolist()))),
    ),
]

# ================= HELPERS =================
def _safe_json_list(raw: str) -> List[str]:
    if not isinstance(raw, str):
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if isinstance(x, (str, int, float))]
    except Exception:
        pass
    if "[" in raw and "]" in raw:
        try:
            candidate = raw[raw.find("[") : raw.rfind("]") + 1]
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if isinstance(x, (str, int, float))]
        except Exception:
            pass
    items = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*\d\.\)\s]+", "", line)
        if line:
            items.append(line)
    return items[:3]


def _parse_tool_json(raw: str):
    """
    Extract JSON even if wrapped in ```json ... ``` or ``` ... ```.
    Returns dict or None.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    # Extract content between code fences
    m = re.findall(r"```(?:json)?\s*(.*?)```", s, flags=re.S | re.I)
    if m:
        s = m[-1].strip()
    if s.lower().startswith("json"):
        s = s[4:].strip()
    try:
        return json.loads(s)
    except Exception:
        return None


def _normalize_to_contract(payload: Any) -> Dict[str, Any]:
    # Already in contract
    if isinstance(payload, dict) and "response_type" in payload:
        return payload

    # Dict payloads
    if isinstance(payload, dict):
        # dict of lists (columns -> values)
        if all(isinstance(v, list) for v in payload.values()) and len(payload) > 0:
            keys = list(payload.keys())
            lengths = [len(payload[k]) for k in keys]
            if len(set(lengths)) == 1:
                rows = [
                    dict(zip(keys, vals))
                    for vals in zip(*[payload[k] for k in keys])
                ]
                return {
                    "response_type": "table",
                    "table": {"columns": keys, "rows": rows},
                }
            # lengths mismatch -> single row of first elements
            row = {
                k: (v[0] if isinstance(v, list) and v else None)
                for k, v in payload.items()
            }
            return {
                "response_type": "table",
                "table": {"columns": list(payload.keys()), "rows": [row]},
            }

        # dict of scalars -> single-row table
        if all(not isinstance(v, (list, dict)) for v in payload.values()):
            return {
                "response_type": "table",
                "table": {"columns": list(payload.keys()), "rows": [payload]},
            }

        # table-like dict
        if "table" in payload and isinstance(payload["table"], dict):
            tbl = payload["table"]
            if "columns" in tbl and "rows" in tbl:
                return {"response_type": "table", "table": tbl}

        # fallback to text
        return {"response_type": "text", "response": json.dumps(payload, ensure_ascii=False)}

    # List payloads
    if isinstance(payload, list):
        if len(payload) > 0 and isinstance(payload[0], dict):
            cols = list({k for row in payload for k in row.keys()})
            return {
                "response_type": "table",
                "table": {"columns": cols, "rows": payload},
            }
        else:
            return {
                "response_type": "table",
                "table": {"columns": ["value"], "rows": [{"value": v} for v in payload]},
            }

    # Scalars or others -> text
    return {"response_type": "text", "response": str(payload)}


def refine_queries(user_input: str) -> List[str]:
    chain = LLMChain(llm=llm, prompt=REFINE_PROMPT)
    raw = chain.run(user_query=user_input)
    refined = _safe_json_list(raw)
    if len(refined) < 3:
        fallback = [
            "Rephrase your query to mention a specific column or brand",
            "Ask for a list of values instead of general info",
            "Specify whether you want counts, details, or descriptions",
        ]
        refined.extend(fallback[: 3 - len(refined)])
    return refined[:3]


def get_memory(session_id: str) -> CombinedMemory:
    if session_id not in SESSION_MEMORIES:
        summary_memory = ConversationSummaryMemory(
            llm=llm, memory_key="chat_history_summary"
        )
        buffer_memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        SESSION_MEMORIES[session_id] = CombinedMemory(
            memories=[summary_memory, buffer_memory]
        )
    return SESSION_MEMORIES[session_id]


def get_rag_chain(session_id: str) -> ConversationalRetrievalChain:
    if session_id in SESSION_RAG_CHAINS:
        return SESSION_RAG_CHAINS[session_id]
    if vectorstore is None:
        raise RuntimeError("Vector store not built. Call /ingest first.")
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": RAG_TOP_K}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=False,
    )
    SESSION_RAG_CHAINS[session_id] = chain
    return chain


def build_faiss_index(df_in: pd.DataFrame) -> FAISS:
    docs: List[Document] = []
    # Column metadata docs
    for col in df_in.columns:
        col_data = df_in[col].dropna()
        col_type = str(df_in[col].dtype)
        sample_values = (
            col_data.sample(min(5, len(col_data))).astype(str).tolist()
            if not col_data.empty
            else []
        )
        meta_summary = f"Column: {col}, Type: {col_type}, Sample: {sample_values}"
        docs.append(
            Document(
                page_content=meta_summary, metadata={"type": "column", "col_name": col}
            )
        )
    # Row chunk docs
    for start in range(0, len(df_in), ROW_CHUNK_SIZE):
        end = min(start + ROW_CHUNK_SIZE, len(df_in))
        chunk = df_in.iloc[start:end]
        text_repr = chunk.to_csv(index=False)
        docs.append(
            Document(
                page_content=text_repr,
                metadata={"type": "rows", "row_range": f"{start}-{end}"},
            )
        )
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    vs = FAISS.from_documents(docs, embeddings)
    return vs


# ---------- NL → Pandas: codegen + safe execution ----------
def _build_chat_context(session_id: str, limit: int = 4) -> str:
    mem = get_memory(session_id)
    lines: List[str] = []
    buf = mem.load_memory_variables({}).get("chat_history") or []
    for m in buf[-limit:]:
        role = getattr(m, "type", "user")
        content = getattr(m, "content", "")
        lines.append(f"{role}: {content}")
    # Append a small last-table preview for grounding (not too big)
    lt = SESSION_LAST_TABLE.get(session_id)
    if lt and lt.get("preview"):
        lines.append("assistant: [last_table_preview]\n" + lt["preview"])
    return "\n".join(lines)


def _schema_snapshot(df_in: pd.DataFrame, max_cols=8, max_vals=5) -> str:
    parts = []
    for c in df_in.columns:
        s = df_in[c]
        if s.dtype == object:
            top = s.dropna().astype(str).value_counts().head(max_vals).index.tolist()
            if top:
                parts.append(f"{c}: {top}")
        if len(parts) >= max_cols:
            break
    return "\n".join(parts)


def _build_value_index(
    df_in: pd.DataFrame, max_cols: int = 6, max_unique: int = 2000
) -> set:
    """
    Build a lowercase set of string values present in the last table
    to detect follow-up references generically.
    """
    values = set()
    cnt = 0
    for c in df_in.columns[:max_cols]:
        s = df_in[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            uniq = s.dropna().astype(str).unique().tolist()
            for v in uniq:
                values.add(v.strip().lower())
                cnt += 1
                if cnt >= max_unique:
                    return values
    return values


def _df_preview_text(df_in: pd.DataFrame, max_rows: int = 10) -> str:
    try:
        return df_in.head(max_rows).to_csv(index=False)
    except Exception:
        return ""


def nl_to_pandas(query: str, session_id: str) -> Dict[str, Any]:
    """
    Return dict with:
      { "response_type": "table", "df": <DataFrame> } OR
      { "response_type": "text", "text": <str> } OR
      { "response_type": "error", "text": <str> }
    """
    if df is None:
        return {"response_type": "error", "text": "No data ingested."}

    columns = ", ".join(map(str, df.columns.tolist()))
    chat_context = _build_chat_context(session_id)
    samples = _schema_snapshot(df)

    code_chain = LLMChain(llm=llm, prompt=NL_TO_PANDAS_PROMPT)
    code = code_chain.run(
        question=query,
        columns=columns,
        chat_context=chat_context
        + (f"\n\nSchema samples:\n{samples}" if samples else ""),
    )

    # Extract code block if model returned fences
    if "```" in code:
        parts = code.split("```")
        code = parts[-2] if len(parts) >= 2 else parts[-1]
        code = re.sub(r"^\s*python\s*", "", code.strip(), flags=re.I)

    # Security: sandboxed exec with minimal safe builtins
    safe_builtins = {
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted,
        "list": list,
        "dict": dict,
        "set": set,
        "any": any,
        "all": all,
        "enumerate": enumerate,
        "range": range,
        "zip": zip,
        "abs": abs,
        "round": round,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
    }
    safe_globals = {"__builtins__": safe_builtins, "pd": pd, "np": np}
    # Provide last_df if available for follow-ups
    last_tab = SESSION_LAST_TABLE.get(session_id, {})
    last_df_obj = last_tab.get("df")
    safe_locals = {"df": df.copy(deep=False)}
    if isinstance(last_df_obj, pd.DataFrame):
        safe_locals["last_df"] = last_df_obj.copy(deep=False)

    try:
        exec(code, safe_globals, safe_locals)
    except Exception as e:
        return {
            "response_type": "error",
            "text": f"Failed to execute generated code: {e}",
        }

    result = safe_locals.get("result_df", None)

    # Normalize results
    if isinstance(result, pd.DataFrame):
        return {"response_type": "table", "df": result}

    if result is None:
        for _, v in list(safe_locals.items()):
            if isinstance(
                v, (pd.DataFrame, pd.Series, list, tuple, dict, str, int, float)
            ):
                result = v
                break

    if isinstance(result, pd.Series):
        return {"response_type": "table", "df": result.to_frame()}
    if isinstance(result, (list, tuple)):
        return {"response_type": "table", "df": pd.DataFrame({"value": list(result)})}
    if isinstance(result, dict):
        return {"response_type": "table", "df": pd.DataFrame([result])}
    if isinstance(result, (str, int, float)):
        return {"response_type": "text", "text": str(result)}

    return {"response_type": "error", "text": "Could not produce a tabular result."}


# ================= TOOLS (Agent-selectable) =================
def _pandas_tool_func(q: str, session_id: str) -> str:
    """
    Returns a JSON string with keys:
      response_type: "table" | "text" | "error" | "table_ref"
      table: { columns, rows }     (when table)
      cache_key, meta              (when table_ref)
      response: str                (when text)
      error: str                   (when error)
    """
    res = nl_to_pandas(q, session_id)

    # Save compact memory summary for continuity
    mem = get_memory(session_id)
    try:
        if res["response_type"] == "table":
            df_out: pd.DataFrame = res["df"]
            mem.save_context(
                {"input": q}, {"output": f"Returned table {df_out.shape}"}
            )
        else:
            mem.save_context(
                {"input": q}, {"output": res.get("text", res["response_type"])}
            )
    except Exception:
        pass

    if res["response_type"] == "table":
        df_out: pd.DataFrame = res["df"]

        # Cache last table for follow-ups
        try:
            SESSION_LAST_TABLE[session_id] = {
                "df": df_out,
                "value_index": _build_value_index(df_out),
                "preview": _df_preview_text(df_out),
            }
        except Exception:
            SESSION_LAST_TABLE[session_id] = {"df": df_out}

        payload = {
            "response_type": "table",
            "table": {
                "columns": list(map(str, df_out.columns.tolist())),
                "rows": json.loads(df_out.to_json(orient="records")),
            },
        }
        # Decide whether to inline or cache to avoid huge tool outputs
        rows_cnt = len(payload["table"]["rows"])
        json_str = json.dumps(payload, ensure_ascii=False)
        if rows_cnt > TABLE_INLINE_ROW_THRESHOLD or len(json_str) > TABLE_INLINE_JSON_CHARS:
            key = f"{session_id}:{uuid4().hex}"
            TABLE_CACHE[key] = payload
            return json.dumps(
                {
                    "response_type": "table_ref",
                    "cache_key": key,
                    "meta": {
                        "rows": rows_cnt,
                        "cols": len(payload["table"]["columns"]),
                    },
                },
                ensure_ascii=False,
            )
        # Small enough → inline
        return json.dumps(payload, ensure_ascii=False)

    if res["response_type"] == "text":
        return json.dumps(
            {"response_type": "text", "response": res.get("text", "")},
            ensure_ascii=False,
        )

    return json.dumps(
        {"response_type": "error", "error": res.get("text", "Execution failed.")},
        ensure_ascii=False,
    )


def _rag_tool_func(q: str, session_id: str) -> str:
    rag_chain = get_rag_chain(session_id)
    out = rag_chain.invoke({"question": q})
    ans = out.get("answer") if isinstance(out, dict) else str(out)

    # Mirror into shared memory for NL->Pandas continuity
    mem = get_memory(session_id)
    try:
        mem.save_context({"input": q}, {"output": ans})
    except Exception:
        pass

    return json.dumps({"response_type": "text", "response": ans}, ensure_ascii=False)


def get_or_create_agent(session_id: str):
    if session_id in SESSION_AGENTS:
        return SESSION_AGENTS[session_id]

    pandas_tool = Tool(
        name="pandas_query",
        func=lambda q, sid=session_id: _pandas_tool_func(q, sid),
        description=(
            "Use for structured dataframe tasks: filtering, sorting, groupby, "
            "aggregations, uniques, joins, and exact lookups."
        ),
        return_direct=True,
    )
    rag_tool = Tool(
        name="rag_query",
        func=lambda q, sid=session_id: _rag_tool_func(q, sid),
        description=(
            "Use for semantic/documentation questions about column meanings, "
            "dataset purpose, notes, and free-text context."
        ),
        return_direct=True,
    )
    tools = [pandas_tool, rag_tool]

    # IMPORTANT: do NOT attach memory to the agent (prevents multi-input-key error).
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs={
            "system_message": (
                "You are a structured data assistant. "
                "You MUST call exactly one tool to answer every question. "
                "Never answer directly without a tool. "
                "Return the tool's output as a plain JSON string (no markdown/"
                "code fences). "
                "Never fabricate data or columns. "
                "If the question asks for table operations, use pandas_query. "
                "If it asks for descriptions or meanings, use rag_query."
            )
        },
    )
    SESSION_AGENTS[session_id] = agent
    return agent


# ================= INGEST =================
@app.route("/ingest", methods=["POST"])
def ingest_route():
    global df, vectorstore, SESSION_MEMORIES, SESSION_RAG_CHAINS, SESSION_AGENTS
    try:
        payload = request.get_json(silent=True) or {}
        data_file = payload.get("data_file") or DATA_FILE

        if data_file.endswith(".csv"):
            df = pd.read_csv(data_file)
        elif data_file.endswith(".xlsx"):
            df = pd.read_excel(data_file)
        else:
            return jsonify({"error": "Unsupported file type. Use CSV or XLSX."}), 400

        t0 = time.time()
        vs = build_faiss_index(df)
        vectorstore = vs
        build_ms = int((time.time() - t0) * 1000)

        # Reset per-session caches as schema changed
        SESSION_MEMORIES.clear()
        SESSION_RAG_CHAINS.clear()
        SESSION_AGENTS.clear()
        SESSION_LAST_TABLE.clear()
        TABLE_CACHE.clear()

        return jsonify(
            {
                "message": f"Data ingested successfully from {data_file}.",
                "rows": len(df),
                "columns": df.columns.tolist(),
                "faiss_build_ms": build_ms,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= QUERY (Agent-driven routing) =================
@app.route("/query", methods=["POST"])
def query_route():
    global df, vectorstore
    if df is None or vectorstore is None:
        return jsonify({"error": "No data ingested yet. Call /ingest first."}), 400

    data = request.get_json(force=True)
    user_input = data.get("query")
    session_id = data.get("session_id", "default")

    if not user_input:
        return jsonify({"error": "query is required"}), 400

    # Optional fast path
    for pat, fn in FAST_PATTERNS:
        if pat.search(user_input or ""):
            ans = fn(user_input)
            # Save to memory
            mem = get_memory(session_id)
            try:
                mem.save_context({"input": user_input}, {"output": ans})
            except Exception:
                pass
            return jsonify(
                {
                    "response_type": "text",
                    "response": ans,
                    "route": "fast",
                    "session_id": session_id,
                }
            )

    try:
        # If user mentions a value from the last table, prefer pandas tool (follow-up nudge)
        lt = SESSION_LAST_TABLE.get(session_id)
        if lt and isinstance(lt.get("value_index"), set):
            tokens = re.findall(r"[A-Za-z0-9_]+", (user_input or ""))
            tokens_l = {t.lower() for t in tokens if t}
            pronouns = {"that", "those", "them", "it", "he", "she", "they", "these"}
            if (tokens_l & lt["value_index"]) or (tokens_l & pronouns):
                payload_json = _pandas_tool_func(user_input, session_id)
                try:
                    payload = json.loads(payload_json)
                except Exception:
                    payload = {"response_type": "text", "response": payload_json}

                # Resolve table_ref
                if isinstance(payload, dict) and payload.get("response_type") == "table_ref":
                    key = payload.get("cache_key")
                    cached = TABLE_CACHE.pop(key, None) or {}
                    payload = cached or {
                        "response_type": "error",
                        "error": "Table reference expired or missing.",
                    }

                payload = _normalize_to_contract(payload)
                payload["route"] = "agent"
                payload["session_id"] = session_id
                return jsonify(payload)

        agent = get_or_create_agent(session_id)
        agent_out = agent.invoke({"input": user_input})

        # Extract tool output (may be wrapped or plain)
        raw = (
            agent_out.get("output")
            or agent_out.get("output_text")
            or str(agent_out)
        )

        payload = _parse_tool_json(raw)
        if payload is None:
            # Fallback: if agent didn't produce valid JSON, run a simple heuristic tool choice
            text_l = (user_input or "").lower()
            is_ragish = any(
                k in text_l
                for k in [
                    "meaning",
                    "describe",
                    "what is",
                    "explain",
                    "definition",
                    "context",
                    "documentation",
                    "notes",
                    "purpose",
                    "overview",
                ]
            )
            payload_json = (
                _rag_tool_func(user_input, session_id)
                if is_ragish
                else _pandas_tool_func(user_input, session_id)
            )
            try:
                payload = json.loads(payload_json)
            except Exception:
                payload = {"response_type": "text", "response": str(raw)}

        # Resolve large table references coming from the tool
        if isinstance(payload, dict) and payload.get("response_type") == "table_ref":
            key = payload.get("cache_key")
            cached = TABLE_CACHE.pop(key, None) or {}
            if cached:
                payload = cached
            else:
                payload = {
                    "response_type": "error",
                    "error": "Table reference expired or missing.",
                }

        # Ensure payload matches contract (handles dict/list/scalars)
        payload = _normalize_to_contract(payload)

        # Relaxed refine trigger (avoid penalizing short numerics)
        if payload.get("response_type") == "text":
            txt = payload.get("response", "") or payload.get("error", "")
            low_conf = [
                "not sure",
                "unclear",
                "don't know",
                "possibly",
                "cannot find",
                "ambiguous",
                "unsure",
                "no matching data",
                "not enough information",
                "i don’t know",
                "i don't know",
            ]
            needs_refine = any(k in txt.lower() for k in low_conf) or txt.strip().endswith("?")
            if needs_refine:
                refined = refine_queries(user_input)
                return jsonify(
                    {
                        "response_type": "refine",
                        "refined_queries": refined,
                        "route": "agent",
                        "session_id": session_id,
                    }
                )

        # Pass-through successful payloads (table/text/error)
        payload["route"] = "agent"
        payload["session_id"] = session_id
        return jsonify(payload)

    except Exception as e:
        refined = refine_queries(user_input)
        return jsonify(
            {
                "response_type": "refine",
                "refined_queries": refined,
                "error": str(e),
                "session_id": session_id,
            }
        )


# ================= MAIN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
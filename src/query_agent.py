import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from src.env_vars_utils import get_query_agent
from src.lang_change_utils import chat_completion


class ToolDecision(BaseModel):
    tool: Optional[Literal["llm_chat", "translate"]] = None
    arguments: Optional[Dict[str, Any]] = None
    feedback: Optional[str] = None


AGENT_PROVIDER, AGENT_MODEL, AGENT_TEMPERATURE = get_query_agent()
ROUTING_PROMPT = """Tu tarea es decidir si el usuario quiere usar una herramienta o solo chatear con el modelo. Analiza principalmente el ultimo mensaje del usuario y usa el contexto previo solo si aporta datos ya mencionados.

Herramientas e intenciones:
- translate: traducir texto a otro idioma.
  * Slots obligatorios: "text" (string), "target_language" (string).
  * Slots opcionales: "chunk_size" (int, default 1000), "chunk_overlap" (int, default 0).
- llm_chat: conversar libremente con el modelo.
  * Slots obligatorios: "user" (string) con el ultimo mensaje o la pregunta a responder.

Reglas:
- Si la peticion es de traduccion, devuelve tool "translate" con todos los slots y con isCompleted "true" y reply null. Si falta alguno entonces isCompleted debe ser "false" y en reply pide solo los datos que faltan de forma breve.
- Si la peticion es cualquier otra (pregunta general, charla, intencion desconocida), usa tool "llm_chat" con "user" igual al ultimo mensaje del usuario y isCompleted "true".
- Para "target_language", captura el idioma aunque venga en lenguaje natural o con preposiciones ("a frances", "al ingles", "to French", "en aleman", "en portugues", "in german"). No pidas el idioma si ya se menciono aunque aparezca con acentos, mayusculas/minusculas o preposiciones.
- No inventes slots ni supongas valores ausentes; usa el contexto solo para reciclar datos que el usuario ya dio de forma explicita.
- Responde siempre en el mismo idioma que el ultimo mensaje del usuario.
- Devuelve unicamente JSON estricto sin Markdown ni texto extra con la forma exacta:
{"tool":"translate|llm_chat|null","arguments":{...}|[],"isCompleted":"true|false","reply":"texto o null"}"""


def decide_tool(
    messages: List[dict[str, str]],
) -> Any:
    """
    Decide which tool to call based on the last user message and the prior context.
    """
    history = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    user_last_message = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    prompt = (
        f"Previous context:\n{history}\n\n"
        f"User's last message:\n{user_last_message}"
    )

    raw_response = chat_completion(
        [
            {"role": "system", "content": ROUTING_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=AGENT_MODEL,
        provider=AGENT_PROVIDER,
        temperature=AGENT_TEMPERATURE,
    )

    try:
        parsed = _parse_decision(raw_response)
        normalized = _normalize_decision(parsed, user_last_message, history)
    except Exception:
        return {
            "tool": None,
            "arguments": [],
            "isCompleted": "false",
            "reply": "No pude interpretar la decision del enrutador. Devuelve solo JSON con tool, arguments, isCompleted y reply.",
        }

    return normalized


def _strip_code_fences(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return candidate
        if len(parts) > 1:
            return parts[1].strip()
    return text


def _parse_decision(raw_text: str) -> Dict[str, Any]:
    cleaned = _strip_code_fences(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            return json.loads(snippet)
        raise


def _normalize_decision(parsed: Any, user_last_message: str, history: str) -> Dict[str, Any]:
    base_response = {
        "tool": None,
        "arguments": [],
        "isCompleted": "false",
        "reply": None,
    }

    if not isinstance(parsed, dict):
        base_response["reply"] = "No pude interpretar la decision del enrutador. Devuelve solo JSON con tool, arguments, isCompleted y reply."
        return base_response

    tool = parsed.get("tool")
    arguments = parsed.get("arguments")
    reply = parsed.get("reply")

    if tool == "translate":
        args = arguments if isinstance(arguments, dict) else {}
        missing: list[str] = []

        if not args.get("text"):
            missing.append("text")
        if not args.get("target_language"):
            missing.append("target_language")

        if missing:
            message = reply or f"Faltan datos obligatorios: {', '.join(missing)}."
            return {"tool": None, "arguments": [], "isCompleted": "false", "reply": message}

        args.setdefault("chunk_size", 1000)
        args.setdefault("chunk_overlap", 0)
        return {"tool": "translate", "arguments": args, "isCompleted": "true", "reply": reply}

    if tool == "llm_chat":
        args = arguments if isinstance(arguments, dict) else {}
        if not args.get("user"):
            args["user"] = user_last_message
            
        if not args.get("system"):
            args["system"] = history

        return {"tool": "llm_chat", "arguments": args, "isCompleted": "true", "reply": reply}

    message = reply or "No pude determinar si deseas traducir o chatear. Indica la accion y los datos necesarios."
    return {"tool": None, "arguments": [], "isCompleted": "false", "reply": message}

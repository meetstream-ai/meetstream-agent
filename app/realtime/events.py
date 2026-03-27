"""Map Agents SDK session events to JSON payloads and MeetStream-friendly tool text."""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, Optional

from typing_extensions import assert_never

from agents.realtime import RealtimeSessionEvent


async def serialize_session_event(event: RealtimeSessionEvent) -> Dict[str, Any]:
    """Stable JSON-serializable dict for UI mirroring and internal routing."""
    base_event: Dict[str, Any] = {"type": event.type}
    if event.type == "agent_start":
        base_event["agent"] = event.agent.name
    elif event.type == "agent_end":
        base_event["agent"] = event.agent.name
    elif event.type == "handoff":
        base_event["from"] = event.from_agent.name
        base_event["to"] = event.to_agent.name
    elif event.type == "tool_start":
        base_event["tool"] = event.tool.name
    elif event.type == "tool_end":
        base_event["tool"] = event.tool.name
        base_event["output"] = str(event.output)
    elif event.type == "audio":
        base_event["audio"] = base64.b64encode(event.audio.data).decode("utf-8")
    elif event.type == "audio_interrupted":
        pass
    elif event.type == "audio_end":
        pass
    elif event.type == "history_updated":
        base_event["history"] = [item.model_dump(mode="json") for item in event.history]
    elif event.type == "history_added":
        pass
    elif event.type == "guardrail_tripped":
        base_event["guardrail_results"] = [{"name": r.guardrail.name} for r in event.guardrail_results]
    elif event.type == "raw_model_event":
        base_event["raw_model_event"] = {"type": event.data.type}
    elif event.type == "error":
        base_event["error"] = getattr(event, "error", "Unknown error")
    elif event.type == "input_audio_timeout_triggered":
        pass
    else:
        assert_never(event)
    return base_event


def format_tool_end_for_meeting_chat(tool_name: str | None, raw_output: Any) -> Optional[str]:
    """
    Turn tool output into a single chat line for MeetStream when we can do better than raw JSON
    (e.g. Canva design links).
    """
    try:
        parsed: Any = None
        if isinstance(raw_output, str):
            parsed = json.loads(raw_output)
        elif isinstance(raw_output, dict):
            parsed = raw_output
        if not isinstance(parsed, dict):
            return None
        job = parsed.get("job")
        if not isinstance(job, dict):
            return None
        result = job.get("result") or {}
        designs = result.get("generated_designs") or []
        links: list[tuple[Any, Any]] = []
        for d in designs:
            if not isinstance(d, dict):
                continue
            url = d.get("url")
            thumb = (d.get("thumbnail") or {}).get("url") if isinstance(d.get("thumbnail"), dict) else None
            if url:
                links.append((url, thumb))
        if not links:
            return None
        header = "Canva designs generated:" if (tool_name and "canva" in tool_name.lower()) else "Designs generated:"
        lines = [header]
        for i, (u, t) in enumerate(links, start=1):
            line = f"{i}. {u}"
            if t:
                line += f" (thumb: {t})"
            lines.append(line)
        return "\n".join(lines)
    except Exception:
        return None


def tool_end_message_or_raw(tool_name: str | None, raw_output: Any) -> str:
    pretty = format_tool_end_for_meeting_chat(tool_name, raw_output)
    if pretty:
        return pretty
    if raw_output is None:
        return ""
    return str(raw_output)

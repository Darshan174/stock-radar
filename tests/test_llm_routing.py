from types import SimpleNamespace
from unittest.mock import MagicMock


def _mock_cfg(task_routes: dict[str, list[str]] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        fallback_models=["openai/glm-4.7", "gemini/gemini-2.5-flash"],
        task_model_routes=task_routes or {},
        zai_api_key=None,
        zai_api_base="https://open.bigmodel.cn/api/coding/paas/v4",
        gemini_api_key=None,
        groq_api_key=None,
    )


def test_task_route_parser():
    from config import Settings

    s = Settings(
        LLM_TASK_ROUTES="analysis=groq/a,openai/b;chat=gemini/c;broken_clause"
    )
    assert s.task_model_routes["analysis"] == ["groq/a", "openai/b"]
    assert s.task_model_routes["chat"] == ["gemini/c"]
    assert "broken_clause" not in s.task_model_routes


def test_analyzer_task_route_prefers_task_models(monkeypatch):
    import agents.analyzer as analyzer_mod

    monkeypatch.setattr(
        analyzer_mod,
        "_cfg",
        _mock_cfg(
            {
                "analysis": [
                    "groq/llama-3.1-70b-versatile",
                    "openai/glm-4.7",
                ]
            }
        ),
        raising=False,
    )

    analyzer = analyzer_mod.StockAnalyzer(
        zai_key="z",
        gemini_key="g",
        groq_key="gr",
        enable_rag=False,
    )
    assert analyzer._models_for_task("analysis")[0] == "groq/llama-3.1-70b-versatile"


def test_analyzer_task_route_filters_unavailable_provider(monkeypatch):
    import agents.analyzer as analyzer_mod

    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    monkeypatch.setattr(
        analyzer_mod,
        "_cfg",
        _mock_cfg(
            {
                "analysis": [
                    "groq/llama-3.1-70b-versatile",
                    "openai/glm-4.7",
                ]
            }
        ),
        raising=False,
    )

    analyzer = analyzer_mod.StockAnalyzer(
        zai_key="z",
        gemini_key="g",
        groq_key=None,
        enable_rag=False,
    )
    route = analyzer._models_for_task("analysis")
    assert route[0] == "openai/glm-4.7"
    assert all(not model.startswith("groq/") for model in route)


def test_chat_task_route_uses_chat_chain(monkeypatch):
    import agents.chat_assistant as chat_mod

    monkeypatch.setattr(
        chat_mod,
        "settings",
        _mock_cfg(
            {
                "chat": [
                    "groq/llama-3.1-70b-versatile",
                    "openai/glm-4.7",
                ]
            }
        ),
        raising=False,
    )

    assistant = chat_mod.StockChatAssistant(
        storage=MagicMock(),
        retriever=MagicMock(),
        zai_key="z",
        gemini_key="g",
        groq_key="gr",
    )
    assert assistant._models_for_task("chat")[0] == "groq/llama-3.1-70b-versatile"
    assert assistant._models_for_task("unknown")[0] == "openai/glm-4.7"

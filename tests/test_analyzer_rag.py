import os
import sys
from types import SimpleNamespace


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.analyzer import StockAnalyzer


class DummyRetriever:
    def __init__(self):
        self.kwargs = None

    def retrieve_context(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            similar_analyses=[{"id": 1}],
            similar_signals=[{"id": 2}],
            relevant_news=[{"id": 3}],
        )

    def format_context_for_prompt(self, context):
        assert context is not None
        return "formatted context"


def test_get_rag_context_uses_indicator_aware_query():
    analyzer = object.__new__(StockAnalyzer)
    analyzer.enable_rag = True
    analyzer.rag_retriever = DummyRetriever()

    context, docs = analyzer._get_rag_context(
        symbol="AAPL",
        mode="intraday",
        quote={"price": 193.21, "change_percent": 1.75},
        indicators={"rsi_14": 62.4, "macd": 1.18},
        news=[],
    )

    assert context == "formatted context"
    assert len(docs) == 3
    assert analyzer.rag_retriever.kwargs["query"] == (
        "AAPL intraday analysis with RSI 62.4 MACD 1.18 price 193.21 change 1.75%"
    )

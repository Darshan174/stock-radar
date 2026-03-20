import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def test_create_embeddings_client_defaults_to_cohere():
    from services.storage import CohereEmbeddings, create_embeddings_client

    client = create_embeddings_client(
        provider="cohere",
        model=None,
        dimension=None,
        cohere_key="cohere-test",
    )

    assert isinstance(client, CohereEmbeddings)
    assert client.provider_name == "cohere"
    assert client.model == "embed-english-v3.0"
    assert client.configured_dimension == 1024


def test_create_embeddings_client_maps_google_default_model():
    from services.storage import GoogleEmbeddings, create_embeddings_client

    client = create_embeddings_client(
        provider="google",
        model="embed-english-v3.0",
        dimension=1024,
        gemini_key="gemini-test",
    )

    assert isinstance(client, GoogleEmbeddings)
    assert client.provider_name == "google"
    assert client.model == "gemini-embedding-001"
    assert client.configured_dimension == 1024


def test_stock_storage_uses_settings_for_embedding_config(monkeypatch):
    import services.storage as storage_mod

    captured: dict[str, object] = {}

    def fake_create_embeddings_client(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            provider_name=kwargs["provider"],
            model=kwargs["model"],
            configured_dimension=kwargs["dimension"],
        )

    monkeypatch.setattr(
        storage_mod,
        "_cfg",
        SimpleNamespace(
            supabase_url="https://example.supabase.co",
            supabase_key="supabase-key",
            embedding_provider="google",
            embedding_model="gemini-embedding-001",
            embedding_dim=1024,
            cohere_api_key=None,
            gemini_api_key="gemini-key",
        ),
        raising=False,
    )
    monkeypatch.setattr(storage_mod, "create_client", lambda url, key: MagicMock())
    monkeypatch.setattr(storage_mod, "create_embeddings_client", fake_create_embeddings_client)

    storage = storage_mod.StockStorage()

    assert storage.embedding_provider == "google"
    assert storage.embedding_model == "gemini-embedding-001"
    assert storage.embedding_dim == 1024
    assert captured == {
        "provider": "google",
        "model": "gemini-embedding-001",
        "dimension": 1024,
        "cohere_key": None,
        "gemini_key": "gemini-key",
    }


def test_embedding_metadata_and_signal_context_helpers():
    from services.storage import EmbeddingResult, StockStorage

    result = EmbeddingResult(
        vector=[0.1, 0.2, 0.3],
        provider="cohere",
        model="embed-english-v3.0",
        dimension=3,
    )

    metadata = StockStorage._embedding_metadata(result, prefix="context_embedding")
    assert metadata == {
        "context_embedding": [0.1, 0.2, 0.3],
        "context_embedding_provider": "cohere",
        "context_embedding_model_name": "embed-english-v3.0",
        "context_embedding_dimension": 3,
    }

    text = StockStorage._build_signal_embedding_text(
        signal_type="entry",
        signal="buy",
        reason="RSI breakout with volume support",
        price_at_signal=182.5,
        importance="high",
    )
    assert "entry buy trading signal" in text
    assert "182.5" in text
    assert "RSI breakout with volume support" in text

"""The suite runs offline with no Hugging Face login (tests/conftest.py): a test whose stub misses can fail, but it can
never write to the real Hub with the developer's token."""
import os


def test_the_hub_is_offline_and_tokenless():
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN")
    home = os.environ.get("HF_HOME", "")
    assert home and not os.path.exists(os.path.join(home, "token"))
    try:
        import huggingface_hub
    except ImportError:
        return
    from huggingface_hub import constants
    assert constants.HF_HUB_OFFLINE
    assert huggingface_hub.get_token() is None

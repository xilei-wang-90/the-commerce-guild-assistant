"""Tests for guild_assistant_web.app (Chainlit handlers)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from guild_assistant_web import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_chainlit():
    """Patch chainlit session and message helpers for every test."""
    session_store: dict[str, object] = {}

    with (
        patch.object(app.cl, "user_session") as mock_session,
        patch.object(app.cl, "Message") as mock_message_cls,
        patch.object(app.cl, "ChatSettings") as mock_settings_cls,
        patch.object(app.cl, "make_async") as mock_make_async,
    ):
        # cl.user_session.get / .set backed by a real dict
        mock_session.get = MagicMock(side_effect=session_store.get)
        mock_session.set = MagicMock(side_effect=session_store.__setitem__)

        # cl.Message(...).send() -> coroutine
        mock_msg_instance = MagicMock()
        mock_msg_instance.send = AsyncMock()
        mock_message_cls.return_value = mock_msg_instance

        # cl.ChatSettings([...]).send() -> returns settings dict
        mock_settings_instance = MagicMock()
        mock_settings_instance.send = AsyncMock(return_value={"mode": "summary"})
        mock_settings_cls.return_value = mock_settings_instance

        # cl.make_async wraps sync fn; in tests just return an AsyncMock
        def _fake_make_async(fn):
            async def _wrapper(*a, **kw):
                return fn(*a, **kw)
            return _wrapper
        mock_make_async.side_effect = _fake_make_async

        yield {
            "session": mock_session,
            "session_store": session_store,
            "Message": mock_message_cls,
            "msg_instance": mock_msg_instance,
            "ChatSettings": mock_settings_cls,
            "settings_instance": mock_settings_instance,
            "make_async": mock_make_async,
        }


# ---------------------------------------------------------------------------
# on_chat_start
# ---------------------------------------------------------------------------

class TestOnChatStart:
    @pytest.mark.asyncio
    async def test_sets_pipeline_and_mode_in_session(self, _mock_chainlit):
        mock_pipeline = MagicMock()
        with patch(
            "guild_assistant_web.app.create_pipeline",
            return_value=mock_pipeline,
        ):
            await app.on_chat_start()

        store = _mock_chainlit["session_store"]
        assert store["pipeline"] is mock_pipeline
        assert store["mode"] == "summary"

    @pytest.mark.asyncio
    async def test_sends_welcome_message(self, _mock_chainlit):
        with patch("guild_assistant_web.app.create_pipeline"):
            await app.on_chat_start()

        _mock_chainlit["msg_instance"].send.assert_called()


# ---------------------------------------------------------------------------
# on_settings_update
# ---------------------------------------------------------------------------

class TestOnSettingsUpdate:
    @pytest.mark.asyncio
    async def test_rebuilds_pipeline_on_mode_change(self, _mock_chainlit):
        _mock_chainlit["session_store"]["mode"] = "summary"
        new_pipeline = MagicMock()
        with patch(
            "guild_assistant_web.app.create_pipeline",
            return_value=new_pipeline,
        ) as mock_create:
            await app.on_settings_update({"mode": "section-reverse-hyde"})

        mock_create.assert_called_once_with("section-reverse-hyde")
        assert _mock_chainlit["session_store"]["pipeline"] is new_pipeline
        assert _mock_chainlit["session_store"]["mode"] == "section-reverse-hyde"

    @pytest.mark.asyncio
    async def test_no_rebuild_when_mode_unchanged(self, _mock_chainlit):
        _mock_chainlit["session_store"]["mode"] = "summary"
        with patch(
            "guild_assistant_web.app.create_pipeline",
        ) as mock_create:
            await app.on_settings_update({"mode": "summary"})

        mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# on_message
# ---------------------------------------------------------------------------

class TestOnMessage:
    @pytest.mark.asyncio
    async def test_calls_pipeline_query(self, _mock_chainlit):
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = "The answer is 42."
        _mock_chainlit["session_store"]["pipeline"] = mock_pipeline

        message = MagicMock()
        message.content = "What is a Yakmel?"

        await app.on_message(message)

        mock_pipeline.query.assert_called_once_with("What is a Yakmel?")

    @pytest.mark.asyncio
    async def test_sends_answer(self, _mock_chainlit):
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = "A fluffy creature."
        _mock_chainlit["session_store"]["pipeline"] = mock_pipeline

        message = MagicMock()
        message.content = "What is a Yakmel?"

        await app.on_message(message)

        _mock_chainlit["Message"].assert_called_with(content="A fluffy creature.")
        _mock_chainlit["msg_instance"].send.assert_called()

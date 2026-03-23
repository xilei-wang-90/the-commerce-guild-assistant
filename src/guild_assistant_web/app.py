"""Chainlit web UI for the Sandrock Knowledge Assistant."""

import chainlit as cl
from chainlit.input_widget import Select
from dotenv import load_dotenv

from guild_assistant_web.pipeline_factory import VALID_MODES, create_pipeline

load_dotenv()

_DEFAULT_MODE = "summary"


@cl.on_chat_start
async def on_chat_start():
    """Initialise the session with the default retrieval mode."""
    settings = await cl.ChatSettings(
        [
            Select(
                id="mode",
                label="Retrieval Mode",
                values=list(VALID_MODES),
                initial_value=_DEFAULT_MODE,
                description="summary = full pages; section-reverse-hyde = per-section chunks",
            ),
        ]
    ).send()

    mode = settings["mode"]
    pipeline = await cl.make_async(create_pipeline)(mode)
    cl.user_session.set("pipeline", pipeline)
    cl.user_session.set("mode", mode)

    await cl.Message(
        content=(
            "Welcome to the **Sandrock Knowledge Assistant**! "
            "Ask me anything about My Time at Sandrock.\n\n"
            f"**Current mode:** `{mode}`\n\n"
            "Change the retrieval mode any time via the settings panel "
            "(gear icon)."
        ),
    ).send()


@cl.on_settings_update
async def on_settings_update(settings):
    """Rebuild the pipeline when the user switches mode."""
    new_mode = settings["mode"]
    old_mode = cl.user_session.get("mode")

    if new_mode == old_mode:
        return

    pipeline = await cl.make_async(create_pipeline)(new_mode)
    cl.user_session.set("pipeline", pipeline)
    cl.user_session.set("mode", new_mode)

    await cl.Message(content=f"Switched to **{new_mode}** mode.").send()


@cl.on_message
async def on_message(message: cl.Message):
    """Run the user's question through the RAG pipeline."""
    pipeline = cl.user_session.get("pipeline")
    answer = await cl.make_async(pipeline.query)(message.content)
    await cl.Message(content=answer).send()

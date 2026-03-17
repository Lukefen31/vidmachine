"""
Chat sidebar — streams messages between user and the Director agent.
Renders past messages and handles new user input.
"""

from __future__ import annotations

import streamlit as st


AGENT_AVATARS = {
    "director": "🎬",
    "ingest": "📥",
    "sourcing": "🌐",
    "assembly": "✂️",
    "export": "📤",
    "qa": "🔍",
    "user": "👤",
    "system": "⚙️",
}


def render_chat(
    messages: list[dict],
    on_send: callable,
    disabled: bool = False,
    awaiting_human: bool = False,
) -> None:
    """
    Render the chat interface.

    Args:
        messages: List of {role, content, agent_name (optional)} dicts
        on_send: Callback called with the user's message string
        disabled: Disable input while agents are running
        awaiting_human: Show a pulsing indicator when Director wants input
    """
    # Message history
    chat_container = st.container()
    with chat_container:
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            agent = msg.get("agent_name", "")

            if role == "user":
                with st.chat_message("user", avatar=AGENT_AVATARS["user"]):
                    st.markdown(content)
            else:
                avatar = AGENT_AVATARS.get(agent, "🤖")
                label = agent.upper() if agent else "AGENT"
                with st.chat_message("assistant", avatar=avatar):
                    st.caption(label)
                    st.markdown(content)

    # Input area
    st.divider()

    if awaiting_human:
        st.info("💬 The Director is waiting for your input…")

    placeholder = (
        "Reply to Director..." if awaiting_human
        else "Tell the Director what to create or change…"
    )

    user_input = st.chat_input(
        placeholder=placeholder,
        disabled=disabled and not awaiting_human,
        key="chat_input",
    )

    if user_input:
        on_send(user_input)


def format_agent_message(agent_name: str, content: str) -> dict:
    """Helper to create a formatted agent message dict for the message list."""
    return {
        "role": "assistant",
        "content": content,
        "agent_name": agent_name,
    }


def format_user_message(content: str) -> dict:
    return {"role": "user", "content": content, "agent_name": "user"}

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_CHAT_URL = os.getenv("BACKEND_CHAT_URL", "http://127.0.0.1:8003/chat")
BACKEND_BASE_URL = os.getenv(
    "BACKEND_BASE_URL",
    BACKEND_CHAT_URL[:-5] if BACKEND_CHAT_URL.endswith("/chat") else BACKEND_CHAT_URL,
)
APP_TITLE = "🦆 CS50 Duck Debugger"


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", uuid.uuid4().hex)
    st.session_state.setdefault("hearts_remaining", 8)
    st.session_state.setdefault("hearts_max", 8)
    st.session_state.setdefault("cooldown_remaining_seconds", 0)
    st.session_state.setdefault("reset_hearts_pending", False)
    st.session_state.setdefault("access_token", None)
    st.session_state.setdefault("whoami", {"auth": False, "email": None, "role": None, "groups": []})
    st.session_state.setdefault("scim_result", None)
    st.session_state.setdefault("last_login_email", None)
    st.session_state.setdefault("last_login_role", None)


def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.session_id = uuid.uuid4().hex


def _build_headers(access_token: Optional[str] = None, override_role: Optional[str] = None) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    if override_role:
        headers["X-User-Role"] = override_role
    return headers


def _serialize_chat_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    serialized: List[Dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        if content:
            serialized.append({"role": role, "content": content})
    return serialized


def call_backend(
    messages: List[Dict[str, Any]],
    mode: str,
    use_rag: bool,
    session_id: str,
    reset_hearts: bool,
    access_token: Optional[str],
    override_role: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {
        "messages": _serialize_chat_messages(messages),
        "mode": mode,
        "use_rag": use_rag,
        "session_id": session_id,
        "reset_hearts": reset_hearts,
    }
    response = requests.post(
        BACKEND_CHAT_URL,
        json=payload,
        headers=_build_headers(access_token=access_token, override_role=override_role),
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def fetch_chunk_excerpt(chunk_id: int, access_token: Optional[str]) -> Dict[str, Any]:
    response = requests.get(
        f"{BACKEND_BASE_URL}/chunks/{chunk_id}",
        headers=_build_headers(access_token=access_token),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _post_json(path: str, payload: Dict[str, Any], access_token: Optional[str] = None) -> Dict[str, Any]:
    response = requests.post(
        f"{BACKEND_BASE_URL}{path}",
        json=payload,
        headers=_build_headers(access_token=access_token),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _patch_json(path: str, payload: Dict[str, Any], access_token: Optional[str] = None) -> Dict[str, Any]:
    response = requests.patch(
        f"{BACKEND_BASE_URL}{path}",
        json=payload,
        headers=_build_headers(access_token=access_token),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _get_json(path: str, access_token: Optional[str] = None) -> Dict[str, Any]:
    response = requests.get(
        f"{BACKEND_BASE_URL}{path}",
        headers=_build_headers(access_token=access_token),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _duck_prefix(text: str) -> str:
    if text.startswith("🦆 "):
        return text
    return f"🦆 {text}"


def _render_citations(citations: List[Dict[str, Any]], access_token: Optional[str]) -> None:
    if not citations:
        return

    st.markdown("**Citations**")
    for citation in citations:
        chunk_id = citation.get("chunk_id")
        source = str(citation.get("source", "unknown"))
        loc = citation.get("loc")
        preview = str(citation.get("preview", ""))

        location_suffix = f", p. {loc}" if loc is not None else ""
        st.markdown(f"- `#{chunk_id}` {source}{location_suffix}")
        if preview:
            st.caption(preview)

        with st.expander(f"View source excerpt (chunk #{chunk_id})"):
            try:
                chunk = fetch_chunk_excerpt(int(chunk_id), access_token=access_token)
                chunk_source = chunk.get("source", "unknown")
                chunk_loc = chunk.get("loc")
                loc_text = f", p. {chunk_loc}" if chunk_loc is not None else ""
                st.caption(f"{chunk_source}{loc_text}")
                st.write(chunk.get("text", ""))
            except requests.HTTPError as exc:
                st.error(f"Could not load excerpt: {exc.response.status_code} {exc.response.text}")
            except requests.RequestException as exc:
                st.error(f"Could not load excerpt: {exc}")


def render_messages(messages: List[Dict[str, Any]], access_token: Optional[str]) -> None:
    for message in messages:
        role = str(message.get("role", "assistant"))
        content = str(message.get("content", ""))
        with st.chat_message(role):
            st.write(_duck_prefix(content) if role == "assistant" else content)

            if role == "assistant":
                mode_used = message.get("mode_used")
                if isinstance(mode_used, str):
                    st.caption(f"mode_used: {mode_used}")
                if message.get("blocked"):
                    st.caption(f"blocked: true ({message.get('block_reason')})")

                citations = message.get("citations")
                if isinstance(citations, list):
                    _render_citations(citations, access_token=access_token)


def parse_backend_reply(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "output_text" in payload and isinstance(payload["output_text"], str):
        citations = payload.get("citations", [])
        if not isinstance(citations, list):
            citations = []
        return {
            "content": payload["output_text"],
            "mode_used": payload.get("mode_used"),
            "citations": citations,
            "blocked": bool(payload.get("blocked", False)),
            "block_reason": payload.get("block_reason"),
            "hearts_remaining": payload.get("hearts_remaining"),
            "hearts_max": payload.get("hearts_max"),
            "cooldown_remaining_seconds": payload.get("cooldown_remaining_seconds", 0),
        }

    if "choices" in payload:
        return {
            "content": str(payload["choices"][0]["message"]["content"]),
            "mode_used": None,
            "citations": [],
            "blocked": False,
            "block_reason": None,
            "hearts_remaining": None,
            "hearts_max": None,
            "cooldown_remaining_seconds": 0,
        }

    raise KeyError("No output_text or choices field in backend response")


def _update_whoami(access_token: Optional[str]) -> None:
    if not access_token:
        st.session_state.whoami = {"auth": False, "email": None, "role": None, "groups": []}
        return
    try:
        st.session_state.whoami = _get_json("/whoami", access_token=access_token)
    except requests.RequestException:
        st.session_state.whoami = {"auth": False, "email": None, "role": None, "groups": []}


def main() -> None:
    init_state()
    st.set_page_config(page_title=APP_TITLE, page_icon="🦆", layout="wide")
    st.title(APP_TITLE)
    st.caption(f"Backend endpoint: {BACKEND_CHAT_URL}")
    _update_whoami(st.session_state.access_token)
    left, spacer, right = st.columns([6, 0.5, 2.5], gap="large")

    with right:
        hearts_remaining = int(st.session_state.hearts_remaining)
        hearts_max = int(st.session_state.hearts_max)
        hearts_visual = ("❤️" * max(0, hearts_remaining)) + ("🖤" * max(0, hearts_max - hearts_remaining))
        st.markdown(f"**Hearts: {hearts_remaining}/{hearts_max}**")
        st.markdown(hearts_visual or "🖤")

        cooldown_remaining = int(st.session_state.cooldown_remaining_seconds)
        if cooldown_remaining > 0:
            minutes_remaining = max(1, cooldown_remaining // 60)
            st.warning(f"Cooldown active: ~{minutes_remaining} min remaining")

        if st.button("Reset hearts", key="reset_hearts_button"):
            st.session_state.reset_hearts_pending = True
        if st.session_state.reset_hearts_pending:
            st.caption("Reset will apply on your next message.")

        with st.expander("🔐 Login (Demo SSO)", expanded=False):
            login_email = st.text_input("Email", value="student@example.com", key="login_email")
            login_role = st.selectbox(
                "Role",
                options=["student", "teacher"],
                key="login_role",
                on_change=clear_chat,
            )
            login_groups = st.multiselect(
                "Groups",
                options=["CS50_Students", "CS50_Staff"],
                default=["CS50_Students"] if login_role == "student" else ["CS50_Staff"],
                key="login_groups",
            )

            if st.button("Login", key="login_button"):
                if (
                    login_email != st.session_state.last_login_email
                    or login_role != st.session_state.last_login_role
                ):
                    clear_chat()
                try:
                    token_response = _post_json(
                        "/auth/dev-token",
                        {"email": login_email, "role": login_role, "groups": login_groups},
                    )
                    st.session_state.access_token = token_response.get("access_token")
                    _update_whoami(st.session_state.access_token)
                    st.session_state.last_login_email = login_email
                    st.session_state.last_login_role = login_role
                    st.rerun()
                except requests.HTTPError as exc:
                    st.error(f"Token request failed: {exc.response.status_code} {exc.response.text}")
                except requests.RequestException as exc:
                    st.error(f"Token request failed: {exc}")

            if st.session_state.access_token and st.button("Logout", key="logout_button"):
                st.session_state.access_token = None
                st.session_state.whoami = {"auth": False, "email": None, "role": None, "groups": []}

        whoami = st.session_state.whoami
        if whoami.get("auth"):
            email = whoami.get("email") or "unknown"
            role = whoami.get("role") or "unknown"
            groups = whoami.get("groups", [])
            groups_summary = ", ".join(groups) if isinstance(groups, list) and groups else "-"
            st.markdown(f"**whoami:** `{email}` · `{role}`")
            st.caption(f"groups: {groups_summary}")
        else:
            st.caption("whoami: anonymous")

        if whoami.get("role") == "teacher":
            with st.expander("🛠 Provisioning (Demo SCIM)", expanded=False):
                group_name = st.text_input("Create group", value="CS50_Students", key="scim_group_name")
                if st.button("Create group", key="scim_create_group"):
                    try:
                        st.session_state.scim_result = _post_json(
                            "/scim/v2/Groups",
                            {"displayName": group_name},
                            access_token=st.session_state.access_token,
                        )
                    except requests.HTTPError as exc:
                        st.session_state.scim_result = {"error": f"{exc.response.status_code} {exc.response.text}"}
                    except requests.RequestException as exc:
                        st.session_state.scim_result = {"error": str(exc)}

                create_user_email = st.text_input(
                    "Create user email",
                    value="new_student@example.com",
                    key="scim_user_email",
                )
                create_user_role = st.selectbox(
                    "Create user role",
                    options=["student", "teacher"],
                    key="scim_user_role",
                )
                create_user_active = st.checkbox("User active", value=True, key="scim_user_active")
                if st.button("Create user", key="scim_create_user"):
                    try:
                        st.session_state.scim_result = _post_json(
                            "/scim/v2/Users",
                            {
                                "email": create_user_email,
                                "active": create_user_active,
                                "role": create_user_role,
                            },
                            access_token=st.session_state.access_token,
                        )
                    except requests.HTTPError as exc:
                        st.session_state.scim_result = {"error": f"{exc.response.status_code} {exc.response.text}"}
                    except requests.RequestException as exc:
                        st.session_state.scim_result = {"error": str(exc)}

                add_user_id = st.text_input("Add user id", value="1", key="scim_add_user_id")
                add_group_id = st.text_input("To group id", value="1", key="scim_add_group_id")
                if st.button("Add user to group", key="scim_add_member"):
                    try:
                        st.session_state.scim_result = _patch_json(
                            f"/scim/v2/Groups/{int(add_group_id)}",
                            {"add": [str(add_user_id)]},
                            access_token=st.session_state.access_token,
                        )
                    except requests.HTTPError as exc:
                        st.session_state.scim_result = {"error": f"{exc.response.status_code} {exc.response.text}"}
                    except requests.RequestException as exc:
                        st.session_state.scim_result = {"error": str(exc)}

                deactivate_user_id = st.text_input("Deactivate user id", value="1", key="scim_deactivate_user_id")
                if st.button("Deactivate user", key="scim_deactivate_user"):
                    try:
                        st.session_state.scim_result = _patch_json(
                            f"/scim/v2/Users/{int(deactivate_user_id)}",
                            {"active": False},
                            access_token=st.session_state.access_token,
                        )
                    except requests.HTTPError as exc:
                        st.session_state.scim_result = {"error": f"{exc.response.status_code} {exc.response.text}"}
                    except requests.RequestException as exc:
                        st.session_state.scim_result = {"error": str(exc)}

                if st.session_state.scim_result is not None:
                    st.caption("Last SCIM result")
                    st.json(st.session_state.scim_result)

        st.markdown("---")
        if whoami.get("auth"):
            role = str(whoami.get("role", "student")).lower()
            mode = "teacher" if role == "teacher" else "student"
            st.caption(f"Role badge: `{mode}` (from token)")
        else:
            mode_label = st.radio("Mode", options=["Student", "Teacher"], horizontal=True, key="mode_selector")
            mode = mode_label.lower()
        use_rag = st.checkbox("Use RAG context", value=False, key="use_rag_checkbox")
        simulate_sso = st.checkbox(
            "Simulate SSO role override",
            value=False,
            key="simulate_sso_checkbox",
            disabled=bool(whoami.get("auth")),
        )
        override_role = None
        if simulate_sso:
            override_role = st.selectbox("SSO role", options=["student", "teacher"], key="sso_role_selector")

    with left:
        with st.container(height=650):
            render_messages(st.session_state.messages, access_token=st.session_state.access_token)

    authenticated_role = st.session_state.whoami.get("role") if st.session_state.whoami.get("auth") else None
    ui_effective_mode = str(authenticated_role).lower() if authenticated_role else mode
    chat_disabled = (
        ui_effective_mode == "student"
        and int(st.session_state.cooldown_remaining_seconds) > 0
        and not bool(st.session_state.reset_hearts_pending)
    )

    with left:
        user_prompt = st.chat_input("Type a message", disabled=chat_disabled)
    if not user_prompt:
        return

    user_message: Dict[str, Any] = {"role": "user", "content": user_prompt}
    st.session_state.messages.append(user_message)
    try:
        reset_hearts = bool(st.session_state.reset_hearts_pending)
        backend_payload = call_backend(
            messages=st.session_state.messages,
            mode=mode,
            use_rag=use_rag,
            session_id=st.session_state.session_id,
            reset_hearts=reset_hearts,
            access_token=st.session_state.access_token,
            override_role=override_role,
        )
        parsed = parse_backend_reply(backend_payload)
        if isinstance(parsed.get("hearts_remaining"), int):
            st.session_state.hearts_remaining = int(parsed["hearts_remaining"])
        if isinstance(parsed.get("hearts_max"), int):
            st.session_state.hearts_max = int(parsed["hearts_max"])
        if isinstance(parsed.get("cooldown_remaining_seconds"), int):
            st.session_state.cooldown_remaining_seconds = int(parsed["cooldown_remaining_seconds"])

        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": parsed["content"],
            "mode_used": parsed["mode_used"],
            "citations": parsed["citations"],
            "blocked": parsed["blocked"],
            "block_reason": parsed["block_reason"],
        }
    except requests.HTTPError as exc:
        assistant_message = {
            "role": "assistant",
            "content": f"Backend error: {exc.response.status_code} {exc.response.text}",
            "mode_used": None,
            "citations": [],
            "blocked": False,
            "block_reason": None,
        }
    except requests.RequestException as exc:
        assistant_message = {
            "role": "assistant",
            "content": f"Request failed: {exc}",
            "mode_used": None,
            "citations": [],
            "blocked": False,
            "block_reason": None,
        }
    except (KeyError, IndexError, TypeError):
        assistant_message = {
            "role": "assistant",
            "content": "Unexpected response format from backend.",
            "mode_used": None,
            "citations": [],
            "blocked": False,
            "block_reason": None,
        }
    finally:
        st.session_state.reset_hearts_pending = False

    st.session_state.messages.append(assistant_message)
    st.rerun()


if __name__ == "__main__":
    main()

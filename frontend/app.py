import streamlit as st
import requests
import json
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="Scientific Document VQA", layout="wide")

st.title("Scientific Document VQA")
st.markdown("Upload up to 3 images and ask questions about them.")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "images_uploaded_for_current_session" not in st.session_state:
    st.session_state.images_uploaded_for_current_session = False

# Session management helpers
def fetch_sessions():
    try:
        res = requests.get(f"{API_URL}/sessions", timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        pass
    return []

def load_session_history(session_id: str):
    try:
        res = requests.get(f"{API_URL}/sessions/{session_id}", timeout=5)
        if res.status_code == 200:
            data = res.json()
            st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in data.get("messages", [])]
            st.session_state.session_id = session_id
            st.session_state.images_uploaded_for_current_session = True
            st.session_state.last_signatures = ()
            st.rerun()
    except Exception as e:
        st.sidebar.error(f"Failed to load session: {e}")

def delete_session(session_id: str):
    try:
        res = requests.delete(f"{API_URL}/sessions/{session_id}", timeout=5)
        if res.status_code == 200:
            if st.session_state.session_id == session_id:
                st.session_state.messages = []
                st.session_state.session_id = None
                st.session_state.images_uploaded_for_current_session = False
                st.session_state.last_signatures = ()
            st.rerun()
    except Exception as e:
        st.sidebar.error(f"Failed to delete session: {e}")

# Sidebar
with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Choose images (max 3)",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key="image_uploader"
    )
    if uploaded_files and len(uploaded_files) > 3:
        st.warning("Maximum 3 images allowed. Only the first 3 will be processed.")

    # Auto-detect if user changed/uploaded a new set of images -> automatically start fresh session
    current_signatures = tuple((f.name, f.size) for f in uploaded_files) if uploaded_files else ()
    if "last_signatures" not in st.session_state:
        st.session_state.last_signatures = current_signatures
    elif st.session_state.last_signatures != current_signatures:
        st.session_state.last_signatures = current_signatures
        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state.images_uploaded_for_current_session = False
        st.rerun()

    uploaded_files = uploaded_files[:3]

    if uploaded_files:
        st.markdown("**Preview:**")
        for f in uploaded_files:
            st.image(f, caption=f.name, use_container_width=True)

    st.divider()
    
    if st.button("➕ New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state.images_uploaded_for_current_session = False
        st.session_state.last_signatures = ()
        st.rerun()

    if st.session_state.session_id:
        st.caption(f"**Current Session:** `{st.session_state.session_id[:12]}...`")

    st.divider()
    st.subheader("Recent Sessions")
    recent_sessions = fetch_sessions()
    if recent_sessions:
        for s in recent_sessions:
            col1, col2 = st.columns([0.8, 0.2])
            is_active = (s["session_id"] == st.session_state.session_id)
            btn_label = f"{'🟢 ' if is_active else ''}{s['title'][:25]}"
            if col1.button(btn_label, key=f"sess_{s['session_id']}", use_container_width=True):
                load_session_history(s["session_id"])
            if col2.button("🗑️", key=f"del_{s['session_id']}"):
                delete_session(s["session_id"])
    else:
        st.caption("No saved conversations yet.")


def render_chat_content(content: str, is_streaming: bool = False, container = None):
    """Renders text with separate collapsible expander for <think>...</think> blocks."""
    ctx = container if container is not None else st
    
    if "<think>" in content:
        if "</think>" in content:
            parts = content.split("</think>", 1)
            think_part = parts[0].replace("<think>", "").strip()
            answer_part = parts[1].strip()
            
            if container is not None:
                with container.container():
                    with st.expander("Thinking", expanded=False):
                        st.markdown(think_part)
                    if answer_part:
                        st.markdown(answer_part + ("▌" if is_streaming else ""))
            else:
                with st.expander("Thinking", expanded=False):
                    st.markdown(think_part)
                if answer_part:
                    st.markdown(answer_part)
        else:
            # Still generating thinking process
            think_part = content.replace("<think>", "").strip()
            if container is not None:
                with container.container():
                    with st.expander("Thinking", expanded=True):
                        st.markdown(think_part + "▌")
            else:
                with st.expander("Thinking", expanded=False):
                    st.markdown(think_part)
    else:
        if container is not None:
            container.markdown(content + ("▌" if is_streaming else ""))
        else:
            st.markdown(content)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_chat_content(message["content"])
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Prepare request payload
        data = {"text": prompt}
        files = []

        # If user uploaded files and they haven't been bound to a session yet, attach files
        if uploaded_files and not st.session_state.images_uploaded_for_current_session:
            for f in uploaded_files:
                f.seek(0)
                files.append(("images", (f.name, f.read(), f.type)))
        elif st.session_state.session_id:
            # Subsequent turns without re-uploading files: send existing session_id
            data["session_id"] = st.session_state.session_id

        try:
            with requests.post(f"{API_URL}/chat", data=data, files=files if files else None, stream=True) as response:
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith("data: "):
                                data_str = decoded_line[6:]
                                if data_str == "[DONE]":
                                    break
                                try:
                                    json_data = json.loads(data_str)
                                    # Update session_id from server
                                    if "session_id" in json_data:
                                        st.session_state.session_id = json_data["session_id"]
                                        if uploaded_files:
                                            st.session_state.images_uploaded_for_current_session = True
                                    
                                    # Stream content
                                    if "content" in json_data:
                                        full_response += json_data["content"]
                                        render_chat_content(full_response, is_streaming=True, container=message_placeholder)
                                    elif "error" in json_data:
                                        st.error(f"Server Error: {json_data['error']}")
                                        break
                                except json.JSONDecodeError:
                                    continue
                    message_placeholder.empty()
                    with message_placeholder.container():
                        render_chat_content(full_response, is_streaming=False)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")

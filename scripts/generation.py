import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from typing import List

# Function to fetch context from external source (implement accordingly)
def fetch_context() -> List[str]:
    """
    Retrieve context chunks from your data source.
    Replace this stub with the actual retrieval logic.
    """
    return []

# Initialize LLM singleton
def get_llm(model_name: str, base_url: str, temperature: float = 0.7):
    if not hasattr(st.session_state, "llm"):
        try:
            st.session_state.llm = OllamaLLM(model=model_name, base_url=base_url, temperature=temperature)
        except Exception as e:
            st.error(f"Error creating LLM: {e}")
            return None
    return st.session_state.llm

# Stream the response from the LLM using provided context and prompt
def generate_response(llm: OllamaLLM, prompt: str, context: List[str]):
    context_text = "\n\n".join(context)
    template = (
        "You are an AI assistant. Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{prompt}\n\n"
        "Answer:"
    )
    prompt_template = PromptTemplate(template=template, input_variables=["context", "prompt"])
    final_prompt = prompt_template.format(context=context_text, prompt=prompt)

    # Stream tokens
    for chunk in llm.stream(final_prompt):
        yield chunk

# Streamlit Chat UI
st.set_page_config(page_title="Contextual Q&A Chat", layout="wide")
st.title("🧠 Contextual Q&A Chat with Ollama LLM")

# Sidebar settings
with st.sidebar:
    st.header("Ollama Model Provider Settings")
    model_name = st.text_input("Model Name", value="llama3.2:1b")
    base_url = st.text_input("Base URL", value="http://localhost:11434")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Display previous messages
for entry in st.session_state.history:
    st.chat_message(entry["role"]).write(entry["message"])

# Capture user input
user_input = st.chat_input("Enter your question...")
if user_input:
    # Record user message
    st.session_state.history.append({"role": "user", "message": user_input})
    st.chat_message("user").write(user_input)

    # Prepare LLM and context
    llm = get_llm(model_name, base_url, temperature)
    if llm is None:
        st.stop()
    context_chunks = fetch_context()

    # Stream assistant response in a single bubble
    st.session_state.history.append({"role": "assistant", "message": ""})
    # Display placeholder bubble
    with st.chat_message("assistant"):
        placeholder = st.empty()
        assistant_msg = ""
        for token in generate_response(llm, user_input, context_chunks):
            assistant_msg += token
            # Update in place without repeating
            placeholder.markdown(assistant_msg)
        # After complete, ensure full message is in history
        st.session_state.history[-1]["message"] = assistant_msg
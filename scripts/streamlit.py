import streamlit as st
import atexit
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import *
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from typing import List
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from typing import List
from sentence_transformers import SentenceTransformer
import torch
from rag import fetch_context, rag_init
from weaviate_utils import host_init
from index_chunks import index_init



##ignore some warnings
import warnings

warnings.filterwarnings("ignore", message=".*no running event loop.*")
warnings.filterwarnings("ignore", message=".*Uncaught app execution.*")

# llm singleton
def get_llm(model_name: str, base_url: str, temperature: float = 0.7):
    # If an llm already exists, check if its config has changed
    if "llm" in st.session_state:
        if (
            st.session_state.current_model_name != model_name
            or st.session_state.current_base_url != base_url
            or st.session_state.current_temperature != temperature
        ):
            # Recreate the LLM with new settings
            try:
                st.session_state.llm = OllamaLLM(
                    model=model_name,
                    base_url=base_url,
                    temperature=temperature
                )
                st.session_state.current_model_name = model_name
                st.session_state.current_base_url = base_url
                st.session_state.current_temperature = temperature
            except Exception as e:
                st.error(f"Error creating LLM: {e}")
                return None
    else:
        # First-time creation
        try:
            st.session_state.llm = OllamaLLM(
                model=model_name,
                base_url=base_url,
                temperature=temperature
            )
            st.session_state.current_model_name = model_name
            st.session_state.current_base_url = base_url
            st.session_state.current_temperature = temperature
        except Exception as e:
            st.error(f"Error creating LLM: {e}")
            return None

    return st.session_state.llm

def get_spark_session():
    if "spark" not in st.session_state:
        try:
            st.session_state.spark = SparkSession.builder \
                .appName("weaviate_deneme") \
                .config("spark.driver.bindAddress", "127.0.0.1") \
                .config("spark.driver.host", "127.0.0.1") \
                .getOrCreate()
        except Exception as e:
            st.error(f"Error creating Spark session: {e}")
            return None
    return st.session_state.spark

def get_rag_model():
    if "rag_model" not in st.session_state:
        try:
            st.session_state.bc_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to('cuda').eval()
        except Exception as e:
            st.error(f"Error creating RAG model: {e}")
            return None
    return st.session_state.bc_model

# response streaming, query template
def generate_response(llm: OllamaLLM, prompt: str, context: List[str]):
    context_text = "\n\n".join(context)

    template = (
        "You are an AI assistant. Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{prompt}\n\n"
        "Answer:"
    )
    prompt_template = PromptTemplate(
        template=template,
        input_variables=["context", "prompt"]
    )
    final_prompt = prompt_template.format(context=context_text, prompt=prompt)

    # Stream tokens
    for chunk in llm.stream(final_prompt):
        yield chunk

# chat ui init
st.set_page_config(page_title="Contextual Q&A Chat", layout="wide")
st.title("🧠 Contextual Q&A Chat with Ollama LLM")

# sidebar
with st.sidebar:
    st.header("Ollama Model Provider Settings")
    model_name = st.selectbox(
        "Model Name",
        ("llama3.2:1b", "mistral"),
        key="ui_model_name"
    )
    base_url = st.text_input(
        "Base URL",
        value="http://localhost:11434",
        key="ui_base_url"
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        key="ui_temperature"
    )


if "initialized" not in st.session_state:

    st.session_state.collection_name = "dist_data"

    _ = get_rag_model()
    _ = get_spark_session()

    host_init(st.session_state)
    rag_init(st.session_state)
    index_init(st.session_state)

    st.session_state.initialized = True
    print("Session initialized")
    torch.cuda.empty_cache()

# chat hist
if "history" not in st.session_state:
    st.session_state.history = []

# prev messsages
for entry in st.session_state.history:
    st.chat_message(entry["role"]).write(entry["message"])

# user input
user_input = st.chat_input("Enter your question...")

if user_input:

    st.session_state.history.append({"role": "user", "message": user_input})
    st.chat_message("user").write(user_input)
    

    llm = get_llm(model_name, base_url, temperature)
    rag_model = get_rag_model()
    spark = get_spark_session()

    if llm is None:
        st.stop()

    if spark is None:
        print("Error creating Spark session")
        st.stop()

    context_info = fetch_context(session_state = st.session_state, query=user_input)
    print("Context info:", context_info)
    ## burada distance ve certainity de var
    # Stream assistant response in a single bubble
    st.session_state.history.append({"role": "assistant", "message": ""})
    with st.chat_message("assistant"):
        placeholder = st.empty()
        assistant_msg = ""
        for token in generate_response(llm, user_input, context_info['rag_text']):
            assistant_msg += token
            placeholder.markdown(assistant_msg)
        # After complete, ensure full message is in history
        st.session_state.history[-1]["message"] = assistant_msg

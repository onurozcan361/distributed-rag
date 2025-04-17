import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from typing import List
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import subprocess
import weaviate
import json
import pandas as pd
import random
from typing import List, Dict, Any
import os
from sentence_transformers import SentenceTransformer
from weaviate.classes.query import MetadataQuery

from rag import fetch_context, rag_init
from weaviate_utils import get_external_ips, host_init
from index_chunks import index_init



##ignore some warnings

import warnings

warnings.filterwarnings("ignore", message=".*no running event loop.*")
warnings.filterwarnings("ignore", message=".*Uncaught app execution.*")






# llm singleton
def get_llm(model_name: str, base_url: str, temperature: float = 0.7):
    if not hasattr(st.session_state, "llm"):
        try:
            st.session_state.llm = OllamaLLM(model=model_name, base_url=base_url, temperature=temperature)
        except Exception as e:
            st.error(f"Error creating LLM: {e}")
            return None
    return st.session_state.llm

def get_spark_session():
    if "spark" not in st.session_state:
        try:
            st.session_state.spark = SparkSession.builder \
                .appName("weaviate_deneme") \
                .config("spark.driver.memory", "8g") \
                .config("spark.executor.memory", "8g") \
                .getOrCreate()
        except Exception as e:
            st.error(f"Error creating Spark session: {e}")
            return None
    return st.session_state.spark

# response streaming, query template
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

# chat ui init
st.set_page_config(page_title="Contextual Q&A Chat", layout="wide")
st.title("🧠 Contextual Q&A Chat with Ollama LLM")

# sidebar
with st.sidebar:
    st.header("Ollama Model Provider Settings")
    model_name = st.text_input("Model Name", value="llama3.2:1b")
    base_url = st.text_input("Base URL", value="http://localhost:11434")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)


if "initialized" not in st.session_state:
    st.session_state.spark = SparkSession.builder \
                .appName("weaviate_deneme") \
                .config("spark.driver.memory", "8g") \
                .config("spark.executor.memory", "8g") \
                .getOrCreate()
            
    st.session_state.bc_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to('cuda').eval()
    st.session_state.collection_name = "dist_data"



    host_init(st.session_state)
    rag_init(st.session_state)
    index_init(st.session_state)




    st.session_state.initialized = True
    print("Session initialized")




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


    spark = get_spark_session()
    llm = get_llm(model_name, base_url, temperature)

    if llm is None:
        st.stop()

    if spark is None:
        print("Error creating Spark session")
        st.stop()

  
            
    context_info = fetch_context(session_state = st.session_state, query=user_input)
    
    ## burada distance ve certainity de var
    # Stream assistant response in a single bubble
    st.session_state.history.append({"role": "assistant", "message": ""})
    # Display placeholder bubble
    with st.chat_message("assistant"):
        placeholder = st.empty()
        assistant_msg = ""
        for token in generate_response(llm, user_input, context_info['rag_text']):
            assistant_msg += token
            # Update in place without repeating
            placeholder.markdown(assistant_msg)
        # After complete, ensure full message is in history
        st.session_state.history[-1]["message"] = assistant_msg
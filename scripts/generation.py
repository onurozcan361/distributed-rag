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
from weaviate_utils import get_external_ips

os.environ["COLLECTION_RETRIEVAL_STRATEGY"] = "LocalOnly"


input_schema = StructType([
    StructField("name", StringType(), True),
    StructField("ip", StringType(), True),
    StructField("port", StringType(), True),
    StructField("query", StringType(), True),
])

output_schema = StructType([
    StructField("rag_text", StringType(), True),
    StructField("certainity", FloatType(), True),
    StructField("distance", FloatType(), True),
])

vector_schema = ArrayType(FloatType())

generate_embedding_udf = F.udf(generate_embedding, vector_schema)
search_weaviate_udf = F.udf(search_weaviate, ArrayType(output_schema))

REPARTITION_NUM = 5



def generate_embedding(query: str) -> list:
    try:
        q_embedding = embedding_model.encode(query).tolist()
        return q_embedding
    except Exception as e:
        print(f"err: {e}")
        return []

def search_weaviate(cluster_name: str, cluster_ip: str, cluster_port: str, grpc_ip:str, query_vector: list) -> dict:
    try:
        with  weaviate.connect_to_custom(    
            http_host=cluster_ip,
            http_port=cluster_port,
            http_secure=False,
            grpc_host=grpc_ip,
            grpc_port=50051,
            grpc_secure=False)    as client:
            
            if not client.is_ready():
                raise Exception("Weaviate instance is not ready.")

            chunks = client.collections.get(f"dist_data_{cluster_name[-1]}")
    
            results = chunks.query.near_vector(
                query_vector,
                limit=1,
                return_metadata=MetadataQuery(distance=True, certainty=True),
            )

            client.close()
        
        del chunks, client

    except Exception as e:
        print(f"Error connecting to Weaviate instance: {e}")
        return []
    
    if not results:
        print("empty results")
        return []
    
    return_arr = []
    for result in results.objects:
        metadata = result.metadata
        distance = metadata.distance
        certainity = metadata.certainty
        rag_text = result.properties['context']
        
        return_arr.append(
            [rag_text, certainity, distance]
        )

        
    return return_arr
   

# rag logic
def fetch_context(query : str) -> List[str]:

    df = spark.createDataFrame([services[service] for service in services if service != "grpc"], input_schema)  ## spark df yarat
    df = df.withColumn("query", F.col("query").cast(StringType())).withColumn("query", F.lit(query))            ## lit ile query'yi ekle
    df = df.withColumn("query_embedding", generate_embedding_udf(F.col("query")))                               ## udf ile embeddingleri al

    df = df.withColumn("result_struct", search_weaviate_udf(
        F.col("name"),
        F.col("ip"),
        F.col("port"),
        F.lit(grpc_ip),
        F.col("query_embedding")
    ))

    df = df.withColumn("exploded_result", F.explode("result_struct")) \
    .withColumn("rag_text", F.col("exploded_result.rag_text")) \
        .withColumn("certainity", F.col("exploded_result.certainity")) \
        .withColumn("distance", F.col("exploded_result.distance")) \
        .select("name", "ip", "port", "rag_text", "certainity", "distance") \

    df = df.orderBy("certainity", ascending=False).limit(1)
    result = df.select("rag_text", "certainity", "distance").first()

    return_dict = {
        "rag_text": result.rag_text,
        "certainity": result.certainity,
        "distance": result.distance
    }

    return return_dict


# llm singleton
def get_llm(model_name: str, base_url: str, temperature: float = 0.7):
    if not hasattr(st.session_state, "llm"):
        try:
            st.session_state.llm = OllamaLLM(model=model_name, base_url=base_url, temperature=temperature)
        except Exception as e:
            st.error(f"Error creating LLM: {e}")
            return None
    return st.session_state.llm

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







spark = SparkSession.builder.appName("weaviate_deneme").getOrCreate()
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
services = get_external_ips()
grpc_ip = services["grpc"]["ip"]

# chat ui init
st.set_page_config(page_title="Contextual Q&A Chat", layout="wide")
st.title("🧠 Contextual Q&A Chat with Ollama LLM")

# sidebar
with st.sidebar:
    st.header("Ollama Model Provider Settings")
    model_name = st.text_input("Model Name", value="llama3.2:1b")
    base_url = st.text_input("Base URL", value="http://localhost:11434")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)

# chat hist
if "history" not in st.session_state:
    st.session_state.history = []

# prev messsages
for entry in st.session_state.history:
    st.chat_message(entry["role"]).write(entry["message"])

# user input
user_input = st.chat_input("Enter your question...")

if user_input:
    # Record user message
    st.session_state.history.append({"role": "user", "message": user_input})
    st.chat_message("user").write(user_input)

    # Prepare LLM and context
    llm = get_llm(model_name, base_url, temperature)

    if llm is None:
        st.stop()

    context_info = fetch_context(query=user_input)
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
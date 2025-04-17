import streamlit as st
import atexit
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import *
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from typing import List
from indexer import index_chunks
from weaviate_utils import get_external_ips
from weaviate.classes.query import MetadataQuery
from sentence_transformers import SentenceTransformer
import weaviate

client_ips = []
grpc_host = None
collection_name = None
embedding_model = None

client_ip_global = None
grpc_host_global = None
collection_name_global = None
bc_model_global = None



# Create Spark session at startup
st.set_page_config(page_title="Contextual Q&A Chat", layout="wide")
st.title("🧠 Contextual Q&A Chat with Ollama LLM")
try:
    spark = SparkSession.builder \
        .appName("ContextualQ&A") \
        .master("local[*]") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.host", "127.0.0.1") \
        .getOrCreate()
    # Register Spark stop on app shutdown
    atexit.register(lambda: spark.stop())
    st.write("Spark session started.")
except Exception as e:
    st.error(f"Error initializing Spark session: {e}")
    
def init_clients_and_model(delete: bool = False):
    """
    Initialize Weaviate clients for each service and optionally delete existing collections.
    Also prepares the embedding model on CUDA.
    """
    global client_ips, grpc_host, embedding_model

    services = get_external_ips()
    grpc_host = services["grpc"]["ip"]
    # Move model to GPU and set to eval mode
    embedding_model = embedding_model.to('cuda').eval()

    # Clear and repopulate client IP list
    client_ips.clear()
    for service in services.values():
        if service.get('name') != "grpc":
            client_ips.append(service['ip'])
            if delete:
                try:
                    client = weaviate.connect_to_custom(
                        http_host=service['ip'], http_port=8080, http_secure=False,
                        grpc_host=grpc_host, grpc_port=50051, grpc_secure=False
                    )
                    # Delete any existing collections for this partition
                    for i in range(5):
                        client.collections.delete(f"dist_data_{i}")
                    client.close()
                except Exception as e:
                    print(f"Error deleting on {service['ip']}: {e}")



def close_all_clients():
    """
    Close all active Weaviate client connections.
    """
    global client_ips, grpc_host
    for ip in client_ips:
        try:
            client = weaviate.connect_to_custom(
                http_host=ip, http_port=8080, http_secure=False,
                grpc_host=grpc_host, grpc_port=50051, grpc_secure=False
            )
            client.close()
        except Exception as e:
            print(f"Error closing client {ip}: {e}")

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
close_all_clients()
init_clients_and_model(delete=True)
collection_name = "dist_data"

client_ip_global = spark.sparkContext.broadcast(client_ips)
grpc_host_global = spark.sparkContext.broadcast(grpc_host)
collection_name_global = spark.sparkContext.broadcast(collection_name)
bc_model_global = spark.sparkContext.broadcast(embedding_model)

# Function to fetch context from external source (implement accordingly)
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

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
services = get_external_ips()
grpc_ip = services["grpc"]["ip"]
   
# rag logic
def fetch_context(spark :SparkSession, query : str) -> List[str]:
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
        "rag_text": [result.rag_text],
        "certainity": result.certainity,
        "distance": result.distance
    }

    return return_dict
# Run startup tasks: chunk text and index on app load
try:
    # index_chunks will process and index the chunks (optionally using Spark)
    index_chunks(spark=spark, 
                 client_ip_bc=client_ip_global, 
                 grpc_host_bc=grpc_host_global,
                 collection_name_bc=collection_name_global,
                 model_bc=bc_model_global,
                 delete=True, 
                 chunk_size=128, 
                 num_partitions=5, 
                 articles_path="../Articles")
except Exception as e:
    st.error(f"Error during startup processing: {e}")

# Initialize or recreate LLM singleton whenever config changes
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

# Stream the response from the LLM using provided context and prompt
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

# Streamlit Chat UI


# Sidebar settings
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
    context_chunks = fetch_context(spark=spark, query=user_input)

    # Stream assistant response in a single bubble
    st.session_state.history.append({"role": "assistant", "message": ""})
    with st.chat_message("assistant"):
        placeholder = st.empty()
        assistant_msg = ""
        for token in generate_response(llm, user_input, context_chunks['rag_text']):
            assistant_msg += token
            placeholder.markdown(assistant_msg)
        # After complete, ensure full message is in history
        st.session_state.history[-1]["message"] = assistant_msg

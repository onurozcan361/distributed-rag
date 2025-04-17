from typing import List
from pyspark.sql import functions as F
from pyspark.sql.types import *
import weaviate
from typing import List
from weaviate.classes.query import MetadataQuery

from weaviate_utils import get_external_ips
from pyspark.sql.functions import udf


REPARTITION_NUM = 5

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

output_schema_arr = ArrayType(output_schema)
vector_schema = ArrayType(FloatType())


client_ip_global = None
grpc_host_global = None
collection_name_global = None
bc_model_global = None


@udf(returnType=vector_schema)
def generate_embedding(query: str) -> list:
    try:
        q_embedding = bc_model_global.value.encode(query).tolist()
        return q_embedding
    except Exception as e:
        print(f"err: {e}")
        return []


@udf(returnType=output_schema_arr)
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

            chunks = client.collections.get(f"dist_data_{int(cluster_name[-1]) - 1} ")
    
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
   

def rag_init(session_state) -> None:
    global client_ip_global, grpc_host_global, collection_name_global

    spark = session_state.spark

    client_ip_global = spark.sparkContext.broadcast(session_state.client_ips)
    grpc_host_global = spark.sparkContext.broadcast(session_state.grpc_host)
    collection_name_global = spark.sparkContext.broadcast(session_state.collection_name)


def fetch_context(session_state, query : str) -> List[str]:

    spark = session_state.spark
    services = get_external_ips()
    grpc_ip = services["grpc"]["ip"]

    print("spark session -> " + str(spark))

    global client_ip_global, grpc_host_global, collection_name_global, bc_model_global

    bc_model_global = spark.sparkContext.broadcast(session_state.bc_model)

    df = spark.createDataFrame([services[service] for service in services if service != "grpc"], input_schema)  ## spark df yarat
    df = df.withColumn("query", F.col("query").cast(StringType())).withColumn("query", F.lit(query))            ## lit ile query'yi ekle
    df = df.withColumn("query_embedding", generate_embedding(F.col("query")))                               ## udf ile embeddingleri al

    df = df.withColumn("result_struct", search_weaviate(
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

    del df, result, bc_model_global

    return return_dict


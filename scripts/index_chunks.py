import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("weaviate").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)

import weaviate
from sentence_transformers import SentenceTransformer
from weaviate.classes.data import DataObject
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, spark_partition_id
from pyspark.sql import functions as F
import subprocess
import json
import pandas as pd

from text_chunker import chunk_text
from weaviate_utils import get_external_ips

from pyspark.sql.dataframe import DataFrame
import torch
from tqdm import tqdm

client_ips = []
grpc_host = None
collection_name = None
bc_model = None

client_ip_global, grpc_host_global, collection_name_global, bc_model_global = None, None, None, None

buffer_size = 100

def init_clients_and_model(delete: bool = False):
    global client_ips, grpc_host, collection_name, bc_model

    services = get_external_ips()
    grpc_host = services["grpc"]["ip"]
    bc_model = bc_model.to('cuda').eval()

    for service in get_external_ips().values():
        if service['name'] != "grpc":

            client_ips.append(service['ip'])
            print(f"Service IP: {service['ip']}")

            if delete:
                try:
                    client = weaviate.connect_to_custom(
                        http_host=service['ip'],
                        http_port=8080,
                        http_secure=False,
                        grpc_host=grpc_host,
                        grpc_port=50051,
                        grpc_secure=False
                        )

                    for i in range(5):
                        client.collections.delete(f"dist_data_{i}")
                    client.close()
                except Exception as e:
                    print(f"err -> {service['ip']}: {e}")
    
def close_all_clients():
    global client_ips, grpc_host

    for ip in client_ips:
        try:
            client = weaviate.connect_to_custom(
                http_host=ip,
                http_port=8080,
                http_secure=False,
                grpc_host=grpc_host,
                grpc_port=50051,
                grpc_secure=False
                )

            client.close()
        except Exception as e:
            print(f"err -> {ip}: {e}")



def index_batch_udf(partition_id: int, iterator):
    import torch
    from weaviate.classes.data import DataObject

    global client_ip_global, grpc_host_global, collection_name_global, bc_model_global

    print(f"partition ID -> {partition_id}\n")

    if client_ip_global is None:
        print("client_ip_global is None")
        client_ip_global = client_ips
    if grpc_host_global is None:
        print("grpc_host_global is None")
        grpc_host_global = grpc_host

    client = weaviate.connect_to_custom(
            http_host=client_ip_global.value[partition_id],
            http_port=8080,
            http_secure=False,
            grpc_host=grpc_host,
            grpc_port=50051,
            grpc_secure=False
            )

    buffer = []
    batch_rows = []

    model = bc_model_global.value
    

    with torch.no_grad():
        for row in iterator:
            batch_rows.append(row)

            if len(batch_rows) >= buffer_size:
                try:

                    texts = [r['chunk'] for r in batch_rows]
                    print(f"[{partition_id}] batch processing - > of {len(texts)}")
                    embeddings = model.encode(texts, show_progress_bar=True)
                    embeddings = embeddings
                    # embeddings = embeddings.cpu().numpy()

                    for i, row in enumerate(batch_rows):
                        obj = DataObject(
                            properties={"file_name": row['file_name'], "context": row['chunk']},
                            vector=embeddings[i].tolist()
                        )
                        buffer.append(obj)

                    client.collections.get(f"dist_data_{str(partition_id)}").data.insert_many(objects=buffer)
                    print(f"[{partition_id}] Sent batch of {len(buffer)}")

                except Exception as e:
                    print(f"[{partition_id}] err processing one of the batches: {e}")

                buffer.clear()
                batch_rows.clear()

        if batch_rows:
            try:
                texts = [r['chunk'] for r in batch_rows]
                embeddings = model.encode(texts, show_progress_bar=True)
                embeddings = embeddings

                for i, row in enumerate(batch_rows):
                    obj = DataObject(
                        properties={ "file_name": row['file_name'], "context": row['chunk']},
                        vector=embeddings[i].tolist()
                    )
                    buffer.append(obj)

                client.collections.get(f"dist_data{str(partition_id)}").data.insert_many(objects=buffer)
                print(f"[{partition_id}] Sent final batch of {len(buffer)}")

            except Exception as e:
                print(f"[{partition_id}] err final batch: {e}")


        client.close()

        return buffer


if __name__ == "__main__":

    spark = SparkSession.builder \
        .appName("IndexChunks") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.host", "127.0.0.1") \
        .getOrCreate()

    bc_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    collection_name = "dist_data" ## dist_data_{partition_id} olacak


    close_all_clients()
    init_clients_and_model(delete=True)
    

    client_ip_global = spark.sparkContext.broadcast(client_ips)
    grpc_host_global = spark.sparkContext.broadcast(grpc_host)
    collection_name_global = spark.sparkContext.broadcast(collection_name)
    bc_model_global = spark.sparkContext.broadcast(bc_model)
    
    df_chunks = chunk_text(spark, path="../Articles", chunk_size=64)
    df_chunks.cache()
    df_chunks.repartition(len(client_ips)).rdd.mapPartitionsWithIndex(index_batch_udf, preservesPartitioning=True).count() ## map only

    spark.stop()

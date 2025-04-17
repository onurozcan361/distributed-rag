import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("weaviate").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)

import weaviate
from pyspark.sql import SparkSession
import torch


from text_chunker import chunk_text

# Globals for broadcasting
client_ip_global = None
grpc_host_global = None
collection_name_global = None
bc_model_global = None

buffer_size = 100


def index_batch_udf(partition_id: int, iterator):
    """
    UDF for Spark mapPartitionsWithIndex: processes rows in batches,
    computes embeddings, and indexes into the appropriate Weaviate partition.
    """
    global client_ip_global, grpc_host_global, collection_name_global, bc_model_global
    # Resolve broadcasts
    client_ips = client_ip_global.value
    grpc_host = grpc_host_global.value
    model = bc_model_global.value

    # Connect to the correct Weaviate instance for this partition
    client = weaviate.connect_to_custom(
        http_host=client_ips[partition_id], http_port=8080, http_secure=False,
        grpc_host=grpc_host, grpc_port=50051, grpc_secure=False
    )

    buffer = []
    batch_rows = []
    with torch.no_grad():
        for row in iterator:
            batch_rows.append(row)
            if len(batch_rows) >= buffer_size:
                # Process current batch
                texts = [r['chunk'] for r in batch_rows]
                embeddings = model.encode(texts, show_progress_bar=False)
                for i, r in enumerate(batch_rows):
                    buffer.append(
                        {
                            "properties": {"file_name": r['file_name'], "context": r['chunk']},
                            "vector": embeddings[i].tolist()
                        }
                    )
                client.collections.get(f"dist_data_{partition_id}") \
                      .data.insert_many(objects=buffer)
                buffer.clear()
                batch_rows.clear()
        # Final partial batch
        if batch_rows:
            texts = [r['chunk'] for r in batch_rows]
            embeddings = model.encode(texts, show_progress_bar=False)
            for i, r in enumerate(batch_rows):
                buffer.append(
                    {
                        "properties": {"file_name": r['file_name'], "context": r['chunk']},
                        "vector": embeddings[i].tolist()
                    }
                )
            client.collections.get(f"dist_data_{partition_id}") \
                  .data.insert_many(objects=buffer)

    client.close()
    return []  # Spark requires a return, content is side-effect


def index_chunks(
    spark: SparkSession,
    client_ip_bc,
    grpc_host_bc,
    collection_name_bc,
    model_bc,
    articles_path: str = "./Articles",
    chunk_size: int = 64,
    num_partitions: int = 5,
    delete: bool = False
):
    """
    Main entry for chunking and indexing text documents.

    Args:
        spark (SparkSession): Active Spark session.
        articles_path (str): Path to the input documents.
        chunk_size (int): Token or character count per chunk.
        num_partitions (int): Number of parallel partitions.
        delete (bool): If True, delete existing collections before indexing.
    """
    global client_ip_global, grpc_host_global, collection_name_global, bc_model_global

    # Broadcast shared resources
    client_ip_global = client_ip_bc
    grpc_host_global = grpc_host_bc
    collection_name_global = collection_name_bc
    bc_model_global = model_bc
    print(client_ip_global.value)
    print(grpc_host_global.value)
    print(collection_name_global.value) 
    print(bc_model_global.value)

    # Chunk and index
    df_chunks = chunk_text(spark, path=articles_path, chunk_size=chunk_size)
    df_chunks.cache()
    df_chunks.repartition(num_partitions) \
             .rdd.mapPartitionsWithIndex(index_batch_udf, preservesPartitioning=True) \
             .count()

    print("Indexing complete.")

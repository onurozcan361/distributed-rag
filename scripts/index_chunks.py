# import warnings
# warnings.filterwarnings("ignore")

# import logging
# logging.basicConfig(level=logging.ERROR)
# logging.getLogger("weaviate").setLevel(logging.ERROR)
# logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
# logging.getLogger("pyspark").setLevel(logging.ERROR)

import weaviate
from weaviate.classes.data import DataObject
from text_chunker import chunk_text
import torch

client_ips = None
grpc_host = None
collection_name = None
bc_model = None

buffer_size = 100


def index_init(session_state) -> None:

    global client_ips, grpc_host, collection_name, bc_model
    spark = session_state.spark

    client_ips = session_state.client_ips
    grpc_host = session_state.grpc_host
    collection_name = session_state.collection_name
    bc_model = session_state.bc_model

    def index_batch_udf(partition_id: int, iterator):
        client = weaviate.connect_to_custom(
                http_host=client_ips[partition_id],
                http_port=8080,
                http_secure=False,
                grpc_host=grpc_host,
                grpc_port=50051,
                grpc_secure=False
                )

        buffer = []
        batch_rows = []

        model = bc_model
        

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
        
    df_chunks = chunk_text(spark, path="../Articles", chunk_size=64)
    df_chunks.repartition(len(client_ips)).rdd.mapPartitionsWithIndex(index_batch_udf, preservesPartitioning=True).count() ## map only


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
import subprocess
import json
import pandas as pd

# text_chunker ve weaviate_utils import (değişmedi)
from text_chunker import chunk_text
from weaviate_utils import get_external_ips

# --- GLOBAL DEĞİŞKENLER (değişmedi) ---
cluster_ips_global = None
grpc_host_global = None
collection_name_global = None

# Bu değişkeni sonradan tanımlayacağız (bc_model)
bc_model = None

def index_partition_udf(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Her partide (partition) yer alan veriyi Pandas DataFrame olarak işleyip,
    ilgili Weaviate cluster IP'sine bağlı client üzerinden verileri embed edip indexler.
    İşlem sonunda o partide indexlenen döküman sayısını dönen DataFrame oluşturur.
    """
    global cluster_ips_global, grpc_host_global, collection_name_global
    global bc_model  # broadcast edilmiş model nesnemiz

    partition_id = int(pdf['partition_id'].iloc[0])
    chosen_ip = cluster_ips_global[partition_id % len(cluster_ips_global)]
    print(f"Partition {partition_id} assigned to Weaviate cluster with IP: {chosen_ip}")

    # Weaviate'a bağlan
    client = weaviate.connect_to_custom(
        http_host=chosen_ip,
        http_port=8080,
        http_secure=False,
        grpc_host=grpc_host_global,
        grpc_port=50051,
        grpc_secure=False
    )

    # --- 1) Broadcast edilmiş model CPU tarafında değer olarak gelecek ---
    # --- 2) Burada GPU'ya yüklüyoruz (to('cuda')) ---
    model = bc_model.value.to('cuda')

    # Koleksiyon (class) kontrolü
    try:
        collection = client.collections.get(collection_name_global)
    except Exception as e:
        print(f"Cluster {chosen_ip}: Koleksiyon {collection_name_global} bulunamadı. Hata: {e}")
        try:
            collection = client.collections.get(collection_name_global)
            print(f"Cluster {chosen_ip}: Koleksiyon {collection_name_global} oluşturuldu.")
        except Exception as create_e:
            print(f"Cluster {chosen_ip}: Koleksiyon oluşturulurken hata: {create_e}")
            collection = None

    indexed_count = 0
    if collection is not None and not pdf.empty:
        texts = pdf['chunk'].tolist()
        file_names = pdf['file_name'].tolist()
        # Embedding hesaplama (GPU'da)
        embeddings = model.encode(texts, show_progress_bar=True)

        # Örnek olarak 100'lük batch'ler halinde ekleme
        BATCH_SIZE = 100
        for i in range(0, len(texts), BATCH_SIZE):
            sub_texts = texts[i : i + BATCH_SIZE]
            sub_file_names = file_names[i : i + BATCH_SIZE]
            sub_embeddings = embeddings[i : i + BATCH_SIZE]

            data_objects = []
            for file_name, text, embedding in zip(sub_file_names, sub_texts, sub_embeddings):
                data_objects.append(
                    DataObject(
                        properties={"context": text, "file_name": file_name},
                        vector=embedding
                    )
                )

            try:
                _uuids = collection.data.insert_many(data_objects)
                indexed_count += len(data_objects)
            except Exception as e:
                print(f"Cluster {chosen_ip}: Indexleme sırasında hata: {e}")
                
        print(f"Cluster {chosen_ip}: Toplam {indexed_count} döküman indexlendi (partition {partition_id}).")
    client.close()
    return pd.DataFrame({
        'partition_id': [partition_id],
        'indexed_count': [indexed_count]
    })

if __name__ == "__main__":
    # Weaviate servislerinin IP bilgilerini alıyoruz.
    ext_ips = get_external_ips(namespace="weaviate")
    grpc_info = ext_ips.pop("grpc")
    grpc_host_global = grpc_info['ip']

    cluster_ips_global = [val['ip'] for val in ext_ips.values() if val['ip'] != "No External IP"]
    if len(cluster_ips_global) < 5:
        print("Uyarı: 5'ten az cluster bulundu, mevcut clusterlar döngüsel olarak kullanılacak.")

    print("Cluster IPs OK.")
    collection_name_global = "dist_data"

    # SparkSession yarat
    spark = SparkSession.builder \
        .appName("IndexChunks") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.host", "127.0.0.1") \
        .getOrCreate()

    # --- 1) SentenceTransformer modelini CPU tarafında (driver) yükle ---
    local_model_cpu = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # --- 2) Spark broadcast ile worker'lara dağıt ---
    bc_model = spark.sparkContext.broadcast(local_model_cpu)

    # chunk_text fonksiyonu, {file_name: str, chunks: ARRAY<STRING>} döndürüyor
    df_chunks = chunk_text()
    print(f"Chunks created: {df_chunks.count()}")

    # Veriyi önce 5 partition'a bölüyoruz, ardından her partiye ait partition ID ekliyoruz.
    exploded_df = df_chunks.select("file_name", explode("chunks").alias("chunk")).repartition(5)
    exploded_df = exploded_df.withColumn("partition_id", spark_partition_id())
    spark.sql("select count(*) from {exploded_df} group by partition_id", exploded_df=exploded_df).show()
    print("Partition ID's added.")
    
    # Her partition'ı Pandas UDF ile işleyerek indexleme gerçekleştiriyoruz.
    result_df = exploded_df.groupBy("partition_id") \
                           .applyInPandas(index_partition_udf, schema="partition_id long, indexed_count long")
    result_df.show()

    # (İstenirse toplam sayıyı toplu alabilirsiniz.)
    # total_indexed = result_df.groupBy().sum("indexed_count").collect()[0][0]
    # print(f"Toplam indexlenen döküman sayısı: {total_indexed}")

    spark.stop()

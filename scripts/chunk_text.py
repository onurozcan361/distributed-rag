from typing import List
from transformers import AutoTokenizer
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from pyspark.sql.functions import udf, col, concat_ws

def read_txt_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def chunk_text_on_tokens(*, text: str, tokenizer, chunk_size: int, chunk_overlap: int) -> List[str]:
    try:
        splits: List[str] = []
        input_ids = tokenizer.encode(text)
        total_tokens = len(input_ids)
        max_model_length = getattr(tokenizer, "model_max_length", 8192)
        if chunk_size > max_model_length:
            chunk_size = max_model_length

        start_idx = 0
        while start_idx < total_tokens:
            end_idx = min(start_idx + chunk_size, total_tokens)
            chunk_ids = input_ids[start_idx:end_idx]
            try:
                splits.append(tokenizer.decode(chunk_ids))
            except Exception as ex:
                print(f"Error decoding tokens: {ex}")
            if end_idx == total_tokens:
                break
            start_idx += chunk_size - chunk_overlap
        return splits
    except Exception as e:
        print(f"Error in chunking text: {e}")
        return []

def get_tokenizer():
    if not hasattr(get_tokenizer, "tokenizer"):
        get_tokenizer.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    return get_tokenizer.tokenizer

def chunk_text_worker(text: str) -> List[str]:
    tokenizer = get_tokenizer()
    return chunk_text_on_tokens(text=text, tokenizer=tokenizer, chunk_size=128, chunk_overlap=20)

# Spark oturumunu başlatıyoruz.
spark = SparkSession.builder \
    .appName("TextChunking") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.driver.host", "127.0.0.1") \
    .getOrCreate()

# ./Articles altındaki klasörlerden .txt dosyalarını okuyarak metinleri topluyoruz.
texts = []
articles_path = "./Articles"
if os.path.isdir(articles_path):
    for folder in os.listdir(articles_path):
        folder_path = os.path.join(articles_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".txt"):
                    file_path = os.path.join(folder_path, file)
                    texts.append({"file_name": file, "text": read_txt_file(file_path)})
else:
    print(f"Directory {articles_path} does not exist")

schema = StructType([
    StructField("file_name", StringType(), True),
    StructField("text", StringType(), True)
])
df = spark.createDataFrame(texts, schema=schema)

# UDF tanımlıyoruz.
chunk_text_udf = udf(chunk_text_worker, ArrayType(StringType()))

# DataFrame'i 1 partition'a bölüp, "text" sütununa UDF uygulayarak "chunks" sütununu ekliyoruz,
# ardından "text" sütununu kaldırıyoruz.
df_result = df.repartition(5).withColumn("chunks", chunk_text_udf(col("text"))).drop("text")
df_result.show()

df_csv = df_result.withColumn("chunks_str", concat_ws("|", col("chunks"))).drop("chunks") 

df_csv.write.mode("overwrite").option("header", "true").csv("./output/result_csv")

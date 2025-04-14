import warnings
warnings.filterwarnings("ignore")  

import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("weaviate").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)

from typing import List
from transformers import AutoTokenizer
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from pyspark.sql.functions import udf, col

# Metin okuma fonksiyonu (değişmedi)
def read_txt_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def chunk_text_on_tokens(
    text: str, 
    tokenizer, 
    chunk_size: int, 
    chunk_overlap: int
) -> List[str]:
    """
    text'i tokenizer encode edildikten sonra chunk'lara ayırıp geri döndürür.
    """
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

def chunk_text():
    # 1) SparkSession oluştur
    spark = SparkSession.builder \
        .appName("TextChunking") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.host", "127.0.0.1") \
        .getOrCreate()

    # 2) Tokenizer'ı driver tarafında yükle, broadcast et
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    bc_tokenizer = spark.sparkContext.broadcast(tokenizer)

    # 3) UDF fonksiyonunu, broadcast değişkenini closure'da kullanacak şekilde tanımla
    def chunk_text_worker(text: str) -> List[str]:
        # Worker tarafında broadcast nesnesine erişiliyor
        _tokenizer = bc_tokenizer.value
        return chunk_text_on_tokens(text=text, tokenizer=_tokenizer, chunk_size=128, chunk_overlap=20)

    chunk_text_udf = udf(chunk_text_worker, ArrayType(StringType()))

    # 4) Klasörlerden .txt dosyalarını okuyarak DataFrame oluştur
    texts = []
    articles_path = "../Articles"
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

    # 5) UDF'yi uygulayarak "chunks" sütununu oluştur, orijinal "text" sütununu drop et
    df_result = df.repartition(5).withColumn("chunks", chunk_text_udf(col("text"))).drop("text")

    return df_result

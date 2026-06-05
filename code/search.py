import json
import os
from typing import List, Dict
from dataclasses import dataclass, field
from pymilvus import MilvusClient
from model_api import compute_embedding, ask_qwen, text_rerank,ask_images,ask_dp
from pprint import pprint as pp
import base64
import mimetypes
import os
from prompt import *

MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"
COLLECTION_NAME = "product_manuals"

client = MilvusClient(
    uri=MILVUS_URI,
    token=MILVUS_TOKEN,
)

def route_and_refine(question:str):
    raw_response = ask_dp(system_prompt=route_and_refine_prompt, user_text=question)
    return raw_response


def search_to_query(question,top_n=5):
    json_str = route_and_refine(question)
    result = json.loads(json_str)
    answer_val = result.get("answer")
    if answer_val in ["None", "", None, "null"]:
        target_doc = result.get("target_doc")
        refined_query = result.get("refined_query")
        print(f"正在检索。目标手册: {target_doc}, 优化查询: {refined_query}")
        query_vector = compute_embedding(refined_query)
        search_res = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            filter=f'doc_name == "{target_doc}"',
            output_fields=["doc_name", "content", "image"], 
            limit=top_n,
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}}
        )
        ifanswer = False
        formatted_results = []
        if search_res and len(search_res[0]) > 0:
            for hits in search_res[0]:
                entity = hits.get("entity")
                formatted_results.append({
                    "content": entity.get("content"),
                    "image": list(entity.get("image", [])), # 确保 JSON 可序列化
                })
        print(f"检索结果: {formatted_results}")
        return  ifanswer,formatted_results
    else:
        ifanswer = True
        answer = result.get("answer")
        print(f"直接回答用户问题，无需检索。回答内容: {answer}")
        return ifanswer, answer
    

if __name__ == "__main__":
    question = "手机如何设置静音模式？"
    ifanswer, result = search_to_query(question)
    print(f"是否直接回答: {ifanswer}")
    pp(result)
 
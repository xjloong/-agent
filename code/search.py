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
    # Clean up common JSON formatting issues
    raw_response = raw_response.strip()
    # Remove markdown code fences if present
    if raw_response.startswith("```"):
        lines = raw_response.split("\n")
        raw_response = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return raw_response


def search_to_query(question,top_n=10):
    json_str = route_and_refine(question)
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}, raw response: {json_str[:200]}")
        # Fallback: try to extract JSON from the response
        import re
        match = re.search(r'\{[^}]+\}', json_str)
        if match:
            try:
                result = json.loads(match.group())
            except:
                result = {"target_doc": "None", "refined_query": "None", "answer": "None"}
        else:
            result = {"target_doc": "None", "refined_query": "None", "answer": "None"}
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
            # Collect results with scores for reranking
            candidates = []
            for hits in search_res[0]:
                entity = hits.get("entity")
                score = hits.get("distance", 0)
                candidates.append({
                    "content": entity.get("content"),
                    "image": list(entity.get("image", [])),
                    "score": score
                })

            # Apply reranking if we have enough candidates
            if len(candidates) > 5:
                try:
                    contents = [c["content"] for c in candidates]
                    rerank_scores = text_rerank(refined_query, contents)
                    # Combine rerank scores with original candidates
                    for i, c in enumerate(candidates):
                        c["rerank_score"] = rerank_scores[i] if i < len(rerank_scores) else 0
                    # Sort by rerank score descending
                    candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
                    print(f"Rerank applied, top scores: {[c['rerank_score'] for c in candidates[:3]]}")
                except Exception as e:
                    print(f"Rerank failed, using original order: {e}")

            # Take top 5 after reranking
            for c in candidates[:5]:
                formatted_results.append({
                    "content": c["content"],
                    "image": c["image"],
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
 
import json
import os
from typing import List, Dict

from pymilvus import MilvusClient
from model_api import compute_embedding, ask_qwen, text_rerank
from pprint import pprint as pp
# TODO: 这里请替换为你实际存放 ask_qwen 函数的模块
# from your_llm_module import ask_qwen 

MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"
COLLECTION_NAME = "product_manuals"

client = MilvusClient(
    uri=MILVUS_URI,
    token=MILVUS_TOKEN,
)

def load_doc_names(json_path: str = "doc_names.json") -> List[str]:
    """读取包含所有产品手册名称的 JSON 文件"""
    if not os.path.exists(json_path):
        print(f"警告: 找不到 {json_path} 文件。请先运行建库脚本。")
        return []
    
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def determine_target_manual(user_query: str, doc_names: List[str]) -> str:
    """使用 Qwen 判断用户问题属于哪个手册"""
    
    # 构造系统提示词，强制要求模型只输出手册名
    system_prompt = f"""
    你是一个意图识别与路由助手。你的任务是根据用户的提问，判断是否需要查阅产品手册？如果是，需要查阅哪个产品手册？
    
    当前可用的手册列表如下：
    {json.dumps(doc_names, ensure_ascii=False)}
    
    请严格遵循以下规则：
    1. 思考哪个手册最有可能包含解答用户问题的信息。
    2. 只输出选中的手册名称字符串（必须与列表中的某个名称完全一致）。
    3. 如果用户的问题与所有手册都不相关，请输出 "None"。
    4. 绝不要输出任何解释、标点符号、或多余的文字。
    """
    
    # 调用你现有的 ask_qwen 函数
    # 注意：为了代码能跑，如果你还没导入真实函数，请使用实际函数替换这里的 mock 逻辑
    target_doc = ask_qwen(system_prompt=system_prompt, user_text=user_query)
    
    # 清理大模型可能输出的空格或换行
    return target_doc.strip()


REWRITE_SYSTEM_PROMPT_TEMPLATE = """
你是一个搜索查询优化专家。你的任务是将用户提问转化为最适合在“特定产品手册”内部检索的关键短语。

当前背景：
用户正在查阅的手册是：{target_doc}

处理规则：
1. **核心指令**：严禁在输出的关键词中包含“{target_doc}”或与之高度相关的产品名称。
2. **去噪**：去除所有礼貌用语、语气助词（如“我想问”、“有没有”等）。
3. **聚焦意图**：仅保留描述具体型号、功能、操作、零件或故障的核心动词和名词组成的关键短语。
4. **简洁性**：只返回优化后的检索关键词，不要输出任何解释。

示例：
手册背景：健身追踪器手册
输入：“我想更换健身追踪器的表带，有其他尺寸可选吗？”
输出：“更换表带尺寸的选项”

手册背景：相机用户手册
输入：“这款相机怎么开启静音拍摄模式？”
输出：“开启静音拍摄模式”

"""

def refine_query(user_query: str, target_doc: str) -> str:
    """
    使用上下文感知的方式优化查询词。
    target_doc: 之前通过 determine_target_manual 得到的手册名称
    """
    
    # 将手册名称注入提示词模板
    system_prompt = REWRITE_SYSTEM_PROMPT_TEMPLATE.format(target_doc=target_doc)
    
    # 调用 LLM
    refined_text = ask_qwen(system_prompt=system_prompt, user_text=user_query)
    
    # 清理格式
    refined_text = refined_text.strip().replace('"', '').replace('“', '').replace('”', '')
    
    print(f"--- 检索优化 ---")
    print(f"目标手册: {target_doc}")
    print(f"优化后关键词: {refined_text}")
    
    return refined_text

def search_in_manual(user_query: str, top_k: int = 3) -> List[Dict]:
    """主搜索流程：大模型路由 -> 向量化 -> Milvus 过滤搜索"""
    
    # 1. 获取所有手册名称
    doc_names = load_doc_names()
    if not doc_names:
        return []

    # 2. 调用大模型判断查找范围
    print("正在思考应查阅哪个手册...")
    target_doc = determine_target_manual(user_query, doc_names)
    
    # 验证 LLM 的输出是否合法
    if target_doc not in doc_names:
        print(f"大模型未能匹配到有效的手册。模型输出为: '{target_doc}'")
        return []
        
    print(f"目标锁定 -> 准备在【{target_doc}】中搜索。")

    # 3. 对用户的查询问题进行优化
    refined_query = refine_query(user_query, target_doc)
    
    # 4. 对优化后的查询进行向量化
    query_vector = compute_embedding(refined_query)

    # 5. 在 Milvus 中进行过滤搜索
    # filter 表达式是关键，它让 Milvus 只在 target_doc 对应的向量中计算相似度
    print("正在执行向量检索...")
    search_res = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        filter=f'doc_name == "{target_doc}"', # 精准过滤机制
        output_fields=["doc_name", "content", "image"], 
        limit=top_k,
        search_params={"metric_type": "COSINE", "params": {"nprobe": 10}}
    )

    # 6. 格式化输出
    formatted_results = []
    
    if search_res and len(search_res[0]) > 0:
        for hits in search_res[0]:
            entity = hits.get("entity")
            
            # 获取图片字段并强制转换为 list
            # 如果 image 字段为空，则返回空列表 []
            raw_images = entity.get("image", [])
            sec_images = list(raw_images) if raw_images else []

            # 严格按照你要求的格式封装
            result_item = {
                # "doc_name": entity.get("doc_name"),
                "content": entity.get("content"),
                "image": sec_images  # 这里现在是标准的 Python list 了
            }
            formatted_results.append(result_item)
            
    return formatted_results


def search_in_manual_rerank(user_query: str, em_top_k: int = 5, rerank_top_k: int = 2) -> List[Dict]:
    """主搜索流程：大模型路由 -> 向量化 -> Milvus 过滤搜索"""
    
    # 1. 获取所有手册名称
    doc_names = load_doc_names()
    if not doc_names:
        return []

    # 2. 调用大模型判断查找范围
    print("正在思考应查阅哪个手册...")
    target_doc = determine_target_manual(user_query, doc_names)
    
    # 验证 LLM 的输出是否合法
    if target_doc not in doc_names:
        print(f"大模型未能匹配到有效的手册。模型输出为: '{target_doc}'")
        return []
        
    print(f"目标锁定 -> 准备在【{target_doc}】中搜索。")

    # 3. 对用户的查询问题进行优化
    refined_query = refine_query(user_query, target_doc)
    
    # 4. 对优化后的查询进行向量化
    query_vector = compute_embedding(refined_query)

    # 5. 在 Milvus 中进行过滤搜索
    # filter 表达式是关键，它让 Milvus 只在 target_doc 对应的向量中计算相似度
    print("正在执行向量检索...")
    search_res = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        filter=f'doc_name == "{target_doc}"', # 精准过滤机制
        output_fields=["doc_name", "content", "image"], 
        limit=em_top_k,
        search_params={"metric_type": "COSINE", "params": {"nprobe": 10}}
    )

    pp(f"初步检索到 {search_res} ")
    if not search_res or len(search_res[0]) == 0:
        return []

    # 提取召回的实体数据
    retrieved_hits = [hit.get("entity") for hit in search_res[0]]
    # 提取纯文本内容列表，用于 Rerank 模型输入
    documents_to_rerank = [hit.get("content") for hit in retrieved_hits]

    # 6. 语义重排 (Rerank)
    # 注意：重排时通常使用【原始查询】或【优化后的查询】均可，
    # 但原始查询往往包含更丰富的语义逻辑，对 Rerank 模型效果更好。
    print(f"正在对 {len(documents_to_rerank)} 条召回结果进行重排...")
    
    # 得到重排后的索引顺序
    reranked_indices = text_rerank(
        query=refined_query, # 使用优化后的查询进行精细比对
        documents=documents_to_rerank, 
        top_n=rerank_top_k       # 最终输出用户需要的 top_k 条
    )

    # 7. 根据重排索引，映射回原始数据格式
    final_results = []
    for idx in reranked_indices:
        entity = retrieved_hits[idx]
        final_results.append({
            "doc_name": entity.get("doc_name"),
            "content": entity.get("content"),
            "image": list(entity.get("image", [])) # 确保 JSON 序列化
        })

    return final_results

    


if __name__ == "__main__":
    # 测试一下功能
    query = "“我想更换健身追踪器的表带，有其他尺寸可选吗？”"
    print(f"\n用户问题: {query}")
    
    results = search_in_manual(query, top_k=3)
    print(results)
    # results = search_in_manual_rerank(query, em_top_k=5, rerank_top_k=2)
    # print("\n--- 检索结果 ---")
    # for idx, res in enumerate(results, 1):
    #     print(f"\n【结果 {idx}】")
    #     print(json.dumps(res, ensure_ascii=False, indent=4))
import os
import re
from typing import List, Dict, Optional, Tuple
import sys 
import json
from docx import Document
from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
)
from model_api import compute_embedding
from txt_split import process_manual_to_chunks


MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"

COLLECTION_NAME = "product_manuals"  # 你可以根据需要修改集合名称
VECTOR_DIM = 1024


client = MilvusClient(
    uri=MILVUS_URI,
    token=MILVUS_TOKEN,
)



def build_milvus_library(chunks: List[Dict], drop_existing: bool = False):
    """
    使用提取的 chunks 构建 Milvus 向量库
    :param chunks: 包含 doc_name, content, image 的字典列表
    :param drop_existing: 是否删除并重建已存在的集合
    """
    # 1. 如果集合已存在且需要重建
    if client.has_collection(COLLECTION_NAME) and drop_existing:
        client.drop_collection(COLLECTION_NAME)
        print(f"已删除旧集合: {COLLECTION_NAME}")

    # 2. 定义 Schema
    # 我们使用 Milvus 的新版 Schema 定义方式，直接在 create_collection 中指定或先创建 schema 对象
    if not client.has_collection(COLLECTION_NAME):
        from pymilvus import CollectionSchema, FieldSchema

        fields = [
            # 主键 ID
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            # 文档名称
            FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=255),
            # 文本内容
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            # 图片名称列表 (使用 Milvus 的 Array 功能，非常适合你的 [img1, img2])
            FieldSchema(name="image", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=100, max_length=100),
            # 向量字段
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
        ]
        
        schema = CollectionSchema(fields, description="产品手册图文 RAG 库")
        
        # 3. 创建索引配置
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",  # 或者使用 HNSW 提高速度
            metric_type="COSINE",   # 推荐使用余弦相似度
            params={"nlist": 128}
        )

        # 创建集合
        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )
        print(f"成功创建新集合: {COLLECTION_NAME}")

    # 4. 批量向量化并准备插入数据
    data_to_insert = []
    print(f"正在处理 {len(chunks)} 条数据并计算向量...")

    for chunk in chunks:
        # 调用你已有的函数计算向量
        embedding = compute_embedding(chunk["content"])
        
        data_to_insert.append({
            "doc_name": chunk["doc_name"],
            "content": chunk["content"],
            "image": chunk["image"],
            "vector": embedding
        })

    # 5. 执行插入操作
    insert_result = client.insert(
        collection_name=COLLECTION_NAME,
        data=data_to_insert
    )

    print(f"向量库构建完成！共插入 {len(data_to_insert)} 条记录。")
    return insert_result

def is_doc_already_imported(doc_name: str) -> bool:
    """
    检查文档是否已经存在于 Milvus 集合中
    """
    # 如果集合还不存在，显然文档没被导入过
    if not client.has_collection(COLLECTION_NAME):
        return False
    
    # 执行查询，检查 doc_name 匹配的数量
    # 注意：Milvus 的 query 需要表达式（filter）
    res = client.query(
        collection_name=COLLECTION_NAME,
        filter=f'doc_name == "{doc_name}"',
        output_fields=["id"],
        limit=1 # 只要找到一个记录就说明已导入
    )
    return len(res) > 0

def process_folder_to_milvus(folder_path: str):
    """
    遍历文件夹，对未导入的 .txt 文件进行增量建库，并生成/更新 doc_names.json
    """
    if not client.has_collection(COLLECTION_NAME):
        print(f"正在初始化集合 {COLLECTION_NAME}...")
        build_milvus_library([], drop_existing=False)

    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    if not files:
        print(f"文件夹 {folder_path} 下没有找到 .txt 文件。")
        return

    # 用于保存当前所有已处理的 doc_name
    all_doc_names = set() 

    for file_name in files:
        doc_name = file_name.replace(".txt", "")
        
        print(f"--- 正在检查: {file_name} ---")
        
        if is_doc_already_imported(doc_name):
            print(f"跳过: 文档 '{doc_name}' 已在数据库中，不再重复导入。")
            all_doc_names.add(doc_name) # 即使跳过，也说明库里有它，加入集合
            continue
        
        print(f"正在处理新文档: {file_name}...")
        try:
            chunks = process_manual_to_chunks(folder_path, file_name)
            
            if not chunks:
                print(f"警告: {file_name} 切分后没有产生任何文本块。")
                continue
                
            build_milvus_library(chunks, drop_existing=False)
            print(f"成功导入: {file_name}")
            all_doc_names.add(doc_name) # 成功导入后加入集合
            
        except Exception as e:
            print(f"处理文件 {file_name} 时发生错误: {e}")

    # ================= 新增逻辑：保存到 JSON =================
    json_path = "code/doc_names.json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            # 将 set 转换为 list 保存，ensure_ascii=False 保证中文正常显示
            json.dump(list(all_doc_names), f, ensure_ascii=False, indent=4)
        print(f"手册目录更新成功，已生成/更新 {json_path}")
    except Exception as e:
        print(f"生成 JSON 文件时失败: {e}")


if __name__ == "__main__":
    # 示例：从文本文件提取 chunks 并构建 Milvus 库
    folder = r"data\KownledgeBase\手册"
    file = "健身追踪器手册.txt"
    
    # print("正在提取文本块...")
    # chunks = process_manual_to_chunks(folder, file)
    
    # print("正在构建 Milvus 向量库...")
    # build_milvus_library(chunks, drop_existing=True)

  
    print("开始执行增量建库任务...")
    process_folder_to_milvus(folder)
    print("所有任务处理完毕。")
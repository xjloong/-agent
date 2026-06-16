import os
import re
import json
from database_built import build_milvus_library
from txt_split import process_manual_to_chunks

def process_english_manual_to_20_chunks(file_path):
    """
    暴力解析版：不再通过 json.loads 解析全文，而是直接定位数组位置，
    手动按结构拆分文本和图片列表。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 直接定位 JSON 数组的起始位置
    start_idx = content.find('[')
    if start_idx == -1:
        print("未找到有效内容")
        return []
    
    # 2. 找到第一个文本段和图片列表的分割点（假设是 ", ["）
    # 这是一个经验值，如果你的文件结构比较标准，它能直接定位到图片数组的开头
    split_pos = content.find('", [')
    if split_pos == -1:
        print("无法自动识别文本与图片的边界")
        return []

    # 提取纯文本部分
    full_text = content[start_idx+2 : split_pos]
    
    # 3. 按 "#" 进行逻辑切分
    sections = full_text.split('#')
    
    # 4. 获取图片列表（简化版：从 split_pos 之后截取，并清理掉末尾的符号）
    # 这样避开了 JSON 内部的斜杠转义问题
    image_str = content[split_pos+4 : content.rfind(']')]
    image_list = [img.strip().replace('"', '') for img in image_str.split(',')]

    chunks = []
    global_img_idx = 0
    current_doc_name = "English_General_Manual"
    last_prefix = ""

    for sec in sections:
        sec = sec.strip()
        if not sec: continue
        
        chunk_content = "# " + sec
        pic_count = chunk_content.count('<PIC>')
        
        sec_images = []
        for _ in range(pic_count):
            if global_img_idx < len(image_list):
                img_name = image_list[global_img_idx]
                sec_images.append(img_name)
                
                # 提取前缀逻辑保持不变
                prefix_match = re.match(r'([A-Za-z0-9]+)_', img_name)
                if prefix_match:
                    prefix = prefix_match.group(1)
                    if prefix != last_prefix:
                        current_doc_name = f"English_{prefix}_Manual"
                        last_prefix = prefix
                global_img_idx += 1
        
        chunks.append({
            "doc_name": current_doc_name,
            "content": chunk_content,
            "image": sec_images
        })
            
    return chunks

def build_english_db(english_file_path):
    print("正在切分并识别英文汇总手册的独立文档边界...")
    chunks = process_english_manual_to_20_chunks(english_file_path)
    
    if not chunks:
        print("切分失败，未能提取到文本块。")
        return

    # 统计并打印识别出了多少个独立的英文文档
    doc_names = set(c["doc_name"] for c in chunks)
    print(f"成功将英文汇总手册切分为以下 {len(doc_names)} 个独立手册:")
    for name in doc_names:
        print(f" - {name}")

    print("\n开始向量化并追加到 Milvus 库中...")
    # drop_existing=False 保证我们是追加数据，不破坏已有的中文库
    build_milvus_library(chunks, drop_existing=False)

    # 将新识别出的英文手册名字追加保存到 JSON 中，供大模型读取
    json_path = "code/doc_names.json"
    try:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                all_doc_names = set(json.load(f))
        else:
            all_doc_names = set()
            
        all_doc_names.update(doc_names)
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(list(all_doc_names), f, ensure_ascii=False, indent=4)
        print(f"手册目录更新成功，已将英文手册目录加入 {json_path}")
    except Exception as e:
        print(f"更新 JSON 文件时失败: {e}")

if __name__ == "__main__":
    # 替换为你的汇总英文手册的实际路径
    file_path = r"data/KownledgeBase/English/汇总英文手册.txt"
    build_english_db(file_path)
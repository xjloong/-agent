import os
import re
import ast
from pprint import pprint as pp
import json

def process_manual_to_chunks(folder_path, file_name, chunk_size=512):
    """
    处理单个产品手册文件，按照 '#' 将文本切分为语义 Chunks。
    """
    file_path = os.path.join(folder_path, file_name)
    doc_name = file_name.replace(".txt", "")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_content = f.read()
        
    # 1. 预处理：过滤掉前缀环境标签，提取核心的 JSON 数组部分
    match = re.search(r'\[\s*".*', raw_content, re.DOTALL)
    if not match:
        print(f"跳过 {file_name}: 未找到有效的内容结构。")
        return []
        
    json_str = match.group(0)
    
    # ================= 修复核心 =================
    # 查找所有 "\"，如果它后面跟的不是合法的 JSON 转义字符 (如 ", \, /, b, f, n, r, t, u)
    # 则将这个 "\" 替换为 "\\" (即双斜杠，将其作为普通文本保留)
    json_str = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', json_str)
    # ==========================================
    
    # 2. 解析 JSON 数据
    try:
        data = json.loads(json_str)
        if not isinstance(data, list) or len(data) < 2:
            return []
            
        full_text = data[0]
        image_list = data[1]
    except json.JSONDecodeError as e:
        print(f"解析 {file_name} 的 JSON 格式失败: {e}")
        return []

    # 3. 按 "#" 进行切分
    sections = full_text.split('#')
    
    chunks = []
    global_img_idx = 0
    
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
            
        # 补回 "#" 号，保持文档原有的标题 Markdown 格式
        content = "# " + sec
        
        # 统计当前 Chunk 下包含了几个 <PIC> 标签
        pic_count = content.count('<PIC>')
        
        # 获取当前章节对应的图片名列表
        sec_images = []
        for _ in range(pic_count):
            if global_img_idx < len(image_list):
                sec_images.append(image_list[global_img_idx])
                global_img_idx += 1
            else:
                print(f"警告 {file_name}: <PIC> 的数量超出了提供的图片列表长度。")

        # 4. 将当前章节作为一个独立的 Chunk 保存
        chunks.append({
            "doc_name": doc_name,
            "content": content,
            "image": sec_images
        })
            
    return chunks



if __name__ == "__main__":
    # 使用示例
    folder = r"D:\Code\agent_desgin\data\KownledgeBase\手册"
    file = "洗碗机手册.txt"
    result = process_manual_to_chunks(folder, file)
    pp(result)

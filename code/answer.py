import json
import re
from typing import List, Tuple
from model_api import ask_images, ask_dp
from search import search_to_query
from prompt import route_and_refine_prompt, qa_system_prompt

def generate_final_answer(question: str, images: List[str] = None) -> Tuple[str, List[str]]:
    """
    处理多模态输入，执行检索并生成最终回答，实现 <PIC> 与图片 ID 的 1:1 精确映射。
    """
    if images is None:
        images = []

    # ==========================================
    # 步骤 1: 图文融合处理
    # ==========================================
    if len(images) > 0:
        print("=> [Step 1] 检测到输入图片，正在调用 ask_images 进行图文意图融合...")
        processed_question = ask_images(images, question)
    else:
        print("=> [Step 1] 无图片输入，保留原问题。")
        processed_question = question

    # ==========================================
    # 步骤 2: 意图路由与检索
    # ==========================================
    print(f"=> [Step 2] 正在分析问题意图并检索: {processed_question}")
    ifanswer, search_result = search_to_query(processed_question)

    returned_images = []

    # ==========================================
    # 步骤 3: 带有图文 ID 锚定机制的答案生成
    # ==========================================
    if ifanswer:
        print("=> [Step 3 结论] 路由判定：直接回答，无需查阅手册。")
        final_answer = search_result
    else:
        print("=> [Step 3 结论] 路由判定：已检索到手册内容，正在调大模型生成最终回答...")
        
        context_texts = []
        for item in search_result:
            content = item.get("content", "")
            img_list = item.get("image", [])
            
            # 【核心逻辑】：将模糊的 <PIC> 临时替换为带有精确ID的 <PIC:Manual_XX>
            if img_list:
                for img_id in img_list:
                    # replace(old, new, 1) 保证每次只替换第一个遇到的 <PIC>，按顺序一一映射
                    content = content.replace("<PIC>", f"<PIC:{img_id}>", 1)
            
            # 容错：如果原文中存在多余的没有对应图片 ID 的 <PIC>，直接洗掉，防止干扰大模型
            content = content.replace("<PIC>", "")
            
            context_texts.append(content)
        
        context_str = "\n---\n".join(context_texts)   

        user_input_prompt = f"【参考手册内容】:\n{context_str}\n\n【用户问题】:\n{processed_question}"
        
        # 调用大模型生成带有 <PIC:图片名> 的原始结果
        raw_answer = ask_dp(system_prompt=qa_system_prompt, user_text=user_input_prompt)

        # ==========================================
        # 步骤 4: 后处理 (提取精准 ID 并洗掉占位符还原)
        # ==========================================
        # 1. 提取大模型真正决定引用的图片 ID（匹配 <PIC:任意内容>）
        extracted_ids = re.findall(r"<PIC:\s*(.*?)\s*>", raw_answer)
        
        # 2. 直接赋值给 returned_images，确保文本中有几个 <PIC>，数组里就有几个 ID，实现完美 1:1 映射
        returned_images = extracted_ids
        
        # 3. 将大模型输出的 <PIC:Manual02_11> 统一替换回赛题要求的标准、干净的 <PIC> 标签
        final_answer = re.sub(r"<PIC:\s*.*?\s*>", "<PIC>", raw_answer)

    return final_answer, returned_images

if __name__ == "__main__":
    # 测试刚才的问题
    ans, imgs = generate_final_answer("人体工学椅手册如何调节靠背？", [])
    print(f"最终回答:\n{ans}\n返回的图片列表: {imgs}\n")
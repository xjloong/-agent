import json
import re
from typing import List, Tuple
from model_api import ask_images, ask_dp
from search import search_to_query
from prompt import route_and_refine_prompt, qa_system_prompt

# ====== 引入 LangChain 核心记忆中间件 ======
try:
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError:
    print("⚠️ 缺少依赖，请先运行: pip install langchain-core")

# 全局字典，充当管理多轮对话的“中间件” (Middleware)
session_store = {}

def get_session_history(session_id: str):
    """获取或创建指定 session_id 的短期记忆"""
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

def generate_final_answer(question: str, images: List[str] = None, session_id: str = "default_session") -> Tuple[str, List[str]]:
    """
    处理多模态输入，执行检索并生成最终回答，实现 <PIC> 与图片 ID 的 1:1 精确映射。
    融入 LangChain 短期记忆中间件，完美支持多轮追问。
    """
    if images is None:
        images = []

    # ==========================================
    # 步骤 0: 提取并格式化短期记忆 (Short-term Memory)
    # ==========================================
    history_obj = get_session_history(session_id)
    history_messages = history_obj.messages
    
    history_str = ""
    if history_messages:
        history_str = "\n【历史对话记录】:\n"
        # 仅提取最近 4 轮（8条）对话作为短期记忆，防止 Token 溢出与幻觉
        for msg in history_messages[-8:]: 
            role = "用户" if isinstance(msg, HumanMessage) else "客服"
            history_str += f"{role}: {msg.content}\n"
        print(f"=> [Memory] 已加载会话 {session_id} 的历史记忆 ({len(history_messages)}条)。")

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
    # 步骤 2: 意图路由与检索 (注入历史记忆)
    # ==========================================
    # 【核心优化】：将历史记录拼接到问题前，让路由模型能看懂“刚才那个”、“它”等代词代指什么
    context_aware_question = f"{history_str}\n【当前问题】:{processed_question}" if history_str else processed_question
    
    print(f"=> [Step 2] 正在分析问题意图并检索: {processed_question}")
    ifanswer, search_result = search_to_query(context_aware_question)

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
            
            if img_list:
                for img_id in img_list:
                    content = content.replace("<PIC>", f"<PIC:{img_id}>", 1)
            
            content = content.replace("<PIC>", "")
            context_texts.append(content)
        
        context_str = "\n---\n".join(context_texts)
        
        # 同样将历史记忆注入到生成最终回答的 Prompt 中
        user_input_prompt = f"{history_str}\n【参考手册内容】:\n{context_str}\n\n【当前问题】:\n{processed_question}"
        
        raw_answer = ask_dp(system_prompt=qa_system_prompt, user_text=user_input_prompt)

        extracted_ids = re.findall(r"<PIC:\s*(.*?)\s*>", raw_answer)
        returned_images = extracted_ids
        final_answer = re.sub(r"<PIC:\s*.*?\s*>", "<PIC>", raw_answer)

    # ==========================================
    # 步骤 4: 更新短期记忆 (保存本轮对话)
    # ==========================================
    history_obj.add_messages([
        HumanMessage(content=processed_question),
        AIMessage(content=final_answer)
    ])

    return final_answer, returned_images

if __name__ == "__main__":
    # 测试记忆功能
    print("\n--- 测试案例: 多轮记忆追问 ---")
    ans1, imgs1 = generate_final_answer("人体工学椅手册如何调节靠背？", [], session_id="user_123")
    print(f"第一轮回答:\n{ans1}\n")
    
    # 注意：第二轮提问中没有明确说“人体工学椅”，但系统通过记忆能知道你在问什么
    ans2, imgs2 = generate_final_answer("我刚刚问了什么", [], session_id="user_123")
    print(f"第二轮(追问)回答:\n{ans2}\n")
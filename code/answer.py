import json
import re
import time
from typing import List, Tuple, Generator, Dict, Any, Optional
from model_api import ask_images, ask_dp
from search import search_to_query
from prompt import route_and_refine_prompt, qa_system_prompt


# ============================================================
# 对话记忆系统：基于摘要的持久记忆
# ============================================================
# 设计思路：
#   - 最近 N 条消息保留原始文本（处理"它"、"刚才那个"等指代）
#   - 超出窗口的历史对话，用 LLM 渐进式压缩为摘要
#   - 摘要随对话推进持续合并更新，实现长期持久记忆
#   - 相比简单截断，摘要记忆不会丢失早期关键信息

# 摘要用的系统提示（轻量、专注）
SUMMARY_SYSTEM_PROMPT = (
    "你是一个专业的对话摘要助手。你的任务是将客服对话压缩为简洁的摘要。\n"
    "要求：\n"
    "1. 保留所有关键事实：产品名称、型号、问题类型、用户具体需求、已提供的解决方案\n"
    "2. 保留重要的数字、步骤编号、错误代码、状态描述\n"
    "3. 丢弃寒暄、礼貌用语、重复内容\n"
    "4. 使用与对话相同的语言输出摘要\n"
    "5. 输出一段连贯的话，不超过 300 字\n"
    "6. 只输出摘要本身，不要加任何前缀或解释"
)


class ConversationMemory:
    """
    基于摘要的对话记忆体。

    - 保留最近 max_recent 条消息作为原始上下文
    - 超出窗口的消息被逐步压缩为摘要
    - 对外提供 get_context() 用于拼入 prompt
    """

    def __init__(self, max_recent: int = 6):
        self.messages: List[Dict[str, str]] = []   # {"role": "user"|"assistant", "content": str}
        self.summary: str = ""                       # 历史对话摘要（渐进累积）
        self.max_recent = max_recent                  # 保留原始消息的窗口大小

    def _trigger_summarization(self) -> bool:
        """判断是否应该触发摘要：超出窗口的消息 >= 2 条"""
        return len(self.messages) > self.max_recent and \
               (len(self.messages) - self.max_recent) >= 2

    def _build_summary_prompt(self, overflow: List[Dict[str, str]]) -> str:
        """构造摘要 prompt，将既有摘要与新溢出对话合并"""
        existing = f"【已有摘要】:\n{self.summary}\n\n" if self.summary else ""

        conversation_text = ""
        for msg in overflow:
            role = "用户" if msg["role"] == "user" else "客服"
            conversation_text += f"{role}: {msg['content']}\n"

        return (
            f"{existing}"
            f"【新增对话】:\n{conversation_text}\n\n"
            f"请将已有摘要（如果有）和新增对话合并为一段新的摘要。"
        )

    def add_turn(self, user_msg: str, assistant_msg: str):
        """
        记录一轮完整对话（用户问 + 客服答），并视需要触发摘要压缩。
        摘要放在记录之后执行，不影响当次响应的延迟。
        """
        self.messages.append({"role": "user", "content": user_msg})
        self.messages.append({"role": "assistant", "content": assistant_msg})

        # 如果溢出足够多，触发摘要压缩
        if self._trigger_summarization():
            overflow = self.messages[:-self.max_recent]
            prompt = self._build_summary_prompt(overflow)
            try:
                new_summary = ask_dp(
                    system_prompt=SUMMARY_SYSTEM_PROMPT,
                    user_text=prompt
                )
                if new_summary and new_summary.strip():
                    self.summary = new_summary.strip()
            except Exception as e:
                print(f"[Memory] 摘要生成失败，保留原始消息: {e}")
                # 降级：不摘要，保留消息（下次再试）
                return

            # 只保留最近 max_recent 条原始消息
            self.messages = self.messages[-self.max_recent:]
            print(f"[Memory] 摘要已更新 ({len(self.summary)} 字符)，保留最近 {len(self.messages)} 条消息。")

    def get_context(self) -> str:
        """
        返回用于注入 prompt 的上下文文本。
        格式：【对话历史摘要】（如果有）+ 【近期对话】（原始消息）
        """
        parts: List[str] = []

        if self.summary:
            parts.append(f"【对话历史摘要】:\n{self.summary}")

        if self.messages:
            recent_text = "【近期对话】:\n"
            for msg in self.messages:
                role = "用户" if msg["role"] == "user" else "客服"
                recent_text += f"{role}: {msg['content']}\n"
            parts.append(recent_text)

        return "\n".join(parts) if parts else ""

    def get_message_count(self) -> int:
        return len(self.messages)

    def has_history(self) -> bool:
        return len(self.messages) > 0 or bool(self.summary)


# ============================================================
# 全局会话存储（使用摘要记忆替代 LangChain）
# ============================================================

# key: session_id, value: ConversationMemory 实例
session_store: Dict[str, ConversationMemory] = {}


def get_session_memory(session_id: str) -> ConversationMemory:
    """获取或创建指定 session_id 的对话记忆体"""
    if session_id not in session_store:
        session_store[session_id] = ConversationMemory(max_recent=6)
    return session_store[session_id]


# ============================================================
# 核心答案生成（同步路径）
# ============================================================

def generate_final_answer(
    question: str,
    images: List[str] = None,
    session_id: str = "default_session"
) -> Tuple[str, List[str]]:
    """
    处理多模态输入，执行检索并生成最终回答。
    融入基于摘要的对话记忆，支持长期持久的多轮对话。
    """
    if images is None:
        images = []

    # ========== 步骤 0: 加载对话记忆 ==========
    memory = get_session_memory(session_id)
    history_context = memory.get_context()
    if history_context:
        print(f"=> [Memory] 已加载会话 {session_id} 的记忆（消息数: {memory.get_message_count()}, "
              f"摘要长度: {len(memory.summary)} 字符）")

    # ========== 步骤 1: 图文融合 ==========
    if len(images) > 0:
        print("=> [Step 1] 检测到输入图片，正在调用 ask_images 进行图文意图融合...")
        processed_question = ask_images(images, question)
    else:
        print("=> [Step 1] 无图片输入，保留原问题。")
        processed_question = question

    # ========== 步骤 2: 意图路由与检索（注入对话记忆） ==========
    context_aware_question = (
        f"{history_context}\n【当前问题】:{processed_question}"
        if history_context else processed_question
    )

    print(f"=> [Step 2] 正在分析问题意图并检索: {processed_question}")
    ifanswer, search_result = search_to_query(context_aware_question)

    returned_images: List[str] = []

    # ========== 步骤 3: 答案生成 ==========
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

        # 将记忆上下文注入最终回答生成的 Prompt
        user_input_prompt = (
            f"{history_context}\n"
            f"【参考手册内容】:\n{context_str}\n\n"
            f"【当前问题】:\n{processed_question}"
        )

        raw_answer = ask_dp(system_prompt=qa_system_prompt, user_text=user_input_prompt)

        extracted_ids = re.findall(r"<PIC:\s*(.*?)\s*>", raw_answer)
        returned_images = extracted_ids
        final_answer = re.sub(r"<PIC:\s*.*?\s*>", "<PIC>", raw_answer)

    # ========== 步骤 4: 更新摘要记忆（记录本轮对话） ==========
    memory.add_turn(processed_question, final_answer)

    return final_answer, returned_images


# ============================================================
# 核心答案生成（SSE 流式路径）
# ============================================================

def generate_with_progress(
    question: str,
    images: List[str] = None,
    session_id: str = "default_session"
) -> Generator[Dict[str, Any], None, None]:
    """
    SSE 进度流式生成器：在每个处理步骤 yield 进度事件，最终 yield 完整结果。
    与 generate_final_answer 共享相同的核心逻辑。
    """
    if images is None:
        images = []

    # ====== Step 0: 加载对话记忆 ======
    memory = get_session_memory(session_id)
    history_context = memory.get_context()
    if history_context:
        yield {"type": "step", "step": "memory", "label": "正在加载历史对话..."}

    # ====== Step 1: 图文融合 ======
    if len(images) > 0:
        yield {"type": "step", "step": "vision", "label": "正在解析图片信息..."}
        processed_question = ask_images(images, question)
    else:
        processed_question = question

    # ====== Step 2: 意图路由 + 检索 ======
    yield {"type": "step", "step": "routing", "label": "正在分析问题意图..."}
    context_aware_question = (
        f"{history_context}\n【当前问题】:{processed_question}"
        if history_context else processed_question
    )
    ifanswer, search_result = search_to_query(context_aware_question)

    returned_images: List[str] = []

    # ====== Step 3: 生成最终答案 ======
    if ifanswer:
        final_answer = search_result
    else:
        yield {"type": "step", "step": "searching", "label": "正在检索产品手册..."}

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
        user_input_prompt = (
            f"{history_context}\n"
            f"【参考手册内容】:\n{context_str}\n\n"
            f"【当前问题】:\n{processed_question}"
        )

        yield {"type": "step", "step": "generating", "label": "正在生成回答..."}
        raw_answer = ask_dp(system_prompt=qa_system_prompt, user_text=user_input_prompt)

        extracted_ids = re.findall(r"<PIC:\s*(.*?)\s*>", raw_answer)
        returned_images = extracted_ids
        final_answer = re.sub(r"<PIC:\s*.*?\s*>", "<PIC>", raw_answer)

    # ====== Step 4: 更新摘要记忆 ======
    memory.add_turn(processed_question, final_answer)

    # ====== 最终结果 ======
    yield {
        "type": "result",
        "data": {
            "answer": final_answer,
            "session_id": session_id,
            "timestamp": int(time.time()),
            "returned_images": returned_images
        }
    }


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    print("\n=== 测试案例: 摘要记忆多轮追问 ===\n")

    # 第一轮：询问人体工学椅
    ans1, imgs1 = generate_final_answer(
        "人体工学椅的靠背怎么调节？",
        [],
        session_id="user_123"
    )
    print(f"[第1轮] 回答:\n{ans1[:200]}...\n")

    # 第二轮：追问（不显式提产品名，靠记忆理解"它"）
    ans2, imgs2 = generate_final_answer(
        "调节时有什么安全注意事项吗？",
        [],
        session_id="user_123"
    )
    print(f"[第2轮] 回答:\n{ans2[:200]}...\n")

    # 第三轮：继续追问（测试摘要是否生效）
    ans3, imgs3 = generate_final_answer(
        "如果调节后还是不舒适怎么办？",
        [],
        session_id="user_123"
    )
    print(f"[第3轮] 回答:\n{ans3[:200]}...\n")

    # 查看记忆状态
    mem = get_session_memory("user_123")
    print(f"\n[记忆状态] 消息数: {mem.get_message_count()}, 摘要长度: {len(mem.summary)} 字符")
    if mem.summary:
        print(f"[摘要内容]:\n{mem.summary}")

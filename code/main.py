from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
import time
from model_api import compute_embedding, text_rerank, ask_qwen, ask_images
from search import search_in_manual
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

# ==========================================
# 1. 初始化 FastAPI 与 鉴权配置
# ==========================================
app = FastAPI(title="多模态客服智能体 API", description="支持图文互补与多轮对话的赛事级 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 允许任何源
    allow_credentials=True,
    allow_methods=["*"], # 允许所有 HTTP 方法 (GET, POST 等)
    allow_headers=["*"], # 允许所有 Header (包括 Authorization)
)

KAFU_API_TOKEN = "sk_customer_20260304"
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """校验 Authorization: Bearer {Token} 格式与有效性"""
    token = credentials.credentials
    if token != KAFU_API_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized: Invalid Token")
    return token

# ==========================================
# 2. 赛题标准数据模型 (Pydantic)
# ==========================================
class ChatRequest(BaseModel):
    question: str = Field(..., description="用户的客服问题字符串")
    images: Optional[List[str]] = Field(default=[], description="Base64 格式图片列表，支持0-3张")
    session_id: Optional[str] = Field(default=None, description="客服会话 ID，用于多轮对话")
    stream: Optional[bool] = Field(default=False, description="是否流式响应，默认同步返回")

# ==========================================
# 3. 简易全局内存：用于多轮对话历史管理
# ==========================================
# 实际生产环境建议用 Redis 或 SQLite，比赛/测试阶段用 dict 即可
session_memory = {}

# ==========================================
# 4. 核心端点路由 (POST /chat)
# ==========================================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest, token: str = Depends(verify_token)):
    # 步骤 A：会话管理 (生成或继承 session_id)
    current_session = request.session_id if request.session_id else f"kf_session_{uuid.uuid4().hex[:8]}"

    # 获取当前会话的历史记录（仅取最近3轮，防止上下文过长）
    history = session_memory.get(current_session, [])
    history_str = "\n".join([f"用户: {msg['user']}\n助手: {msg['assistant']}" for msg in history]) if history else "无历史对话"

    # 默认查询词为用户提问
    augmented_query = request.question 

    # 步骤 B：多模态输入解析
    if request.images:
        if len(request.images) > 3:
            raise HTTPException(status_code=400, detail="最多仅支持上传 3 张图片")
            
        # 调用大模型读图提取意图
        image_intent = ask_images(request.images, request.question)
        
        # 将图片信息与原始问题融合
        if image_intent:
            augmented_query = f"用户原始提问：{request.question}\n基于用户上传的图片，补充关键信息如下：{image_intent}"
            print(f"【多模态意图融合】: {augmented_query}")

    # 步骤 C：知识检索 (RAG)
    # ⚠️ 修复点：这里改用 augmented_query 去进行检索
    retrieved_docs = search_in_manual(augmented_query, top_k=3)

    # 步骤 D：动态 Prompt 组装 (闲聊拦截与思维链)
    if not retrieved_docs:
        # ⚠️ 场景 1：无参考知识（系统判断为闲聊或超出业务范围）
        system_prompt = (
            "你是一个专业的智能客服助手。用户当前的问题无需查阅产品手册。"
            "请结合常识和历史对话自然、得体地回复用户，不要提及'根据提供的知识'等字眼。"
        )
        user_prompt = f"【历史对话】:\n{history_str}\n\n【用户提问】：{request.question}"
        
    else:
        # ⚠️ 场景 2：有参考知识（进入业务解答模式）
        context_texts = []
        available_images = []
        for doc in retrieved_docs:
            context_texts.append(doc['content'])
            if doc.get('image'):
                available_images.extend(doc['image'])
                
        context_str = "\n---\n".join(context_texts)
        
        # 强规则 System Prompt：教导大模型思维链拆解与图片处理
        system_prompt = (
            "你是一个专业的智能客服助手。请严格根据提供的『参考知识』回答用户问题。\n"
            "【核心指令】：\n"
            "1. **问题拆解（思维链）**：如果用户一次提问包含了多个子问题（例如问了发货时间、运费、能否送达等），你必须先理清这些问题，在回答中逐条分点清晰作答，绝不能漏答。\n"
            "2. **图文完美互补**：如果参考知识中出现了 `<PIC>` 标识，请你根据语境，从『可用图片列表』中按顺序选取合适的图片名称，并在最终回答中以 `[图片名称]` 的格式替换掉 `<PIC>`，例如 `[Manual16_51]`。\n"
            "3. **幻觉抑制**：回答必须逻辑清晰，不要凭空捏造手册中没有的数据或知识。"
        )
        
        user_prompt = (
            f"【历史对话】:\n{history_str}\n\n"
            f"【可用图片列表】：{available_images}\n\n"
            f"【参考知识】：\n{context_str}\n\n"
            f"【用户当前提问及背景】：{augmented_query}"
        )
    
    # 步骤 E：生成最终多模态回答
    final_answer = ask_qwen(system_prompt, user_prompt)

    # 步骤 F：更新会话记忆
    if current_session not in session_memory:
        session_memory[current_session] = []
    # 追加本轮对话
    session_memory[current_session].append({"user": request.question, "assistant": final_answer})
    # 始终只保留最近的3轮对话，防止 Token 超出限制
    session_memory[current_session] = session_memory[current_session][-3:]

    # 步骤 G：返回严格符合赛题规范的 JSON
    return {
        "code": 0,
        "msg": "success",
        "data": {
            "answer": final_answer,
            "session_id": current_session,
            "timestamp": int(time.time())
        }
    }
import time
import uuid
from typing import List, Optional
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel, Field
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
# 导入你写好的 RAG 核心流水线
# 假设你的流水线代码文件名为 search_answer.py
from search_answer import RAGPipeline, RAGContext

# ==========================================
# 1. 配置与全局初始化
# ==========================================
app = FastAPI(title="多模态客服智能体 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 实际生产中请替换为具体的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 赛题要求的统一 Token 认证 
KAFU_API_TOKEN = "sk_customer_20260304" # 请根据实际情况修改或放置在环境变量中

# 全局初始化 RAG Pipeline，避免每次请求都重新加载文档名等
pipeline = RAGPipeline()

# ==========================================
# 2. Pydantic 数据模型定义 (严格遵循接口定义说明)
# ==========================================
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="核心输入：用户的客服问题字符串 ")
    images: Optional[List[str]] = Field(default=[], description="Base64 格式图片列表，支持 0-3 张 ")
    session_id: Optional[str] = Field(default=None, description="客服会话 ID ")
    stream: Optional[bool] = Field(default=False, description="是否流式响应 ")

class ChatResponseData(BaseModel):
    answer: str
    session_id: str
    timestamp: int
    # 额外补充：方便前端或评测脚本提取对应的 <PIC> 图片 ID
    returned_images: List[str] = []

class ChatResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: ChatResponseData

# ==========================================
# 3. 核心校验依赖
# ==========================================
def verify_authorization(authorization: str = Header(..., description="Bearer Token认证 ")):
    """校验请求头中的 Authorization 字段 """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format. Must start with 'Bearer '")
    
    token = authorization.split("Bearer ")[1].strip()
    if token != KAFU_API_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API Token")
    return token

def validate_images(images: List[str]):
    """校验图片数量和大小规范 """
    if len(images) > 3:
        raise HTTPException(status_code=400, detail="Images count cannot exceed 3.")
    
    for idx, img_b64 in enumerate(images):
        # 粗略校验 Base64 图片大小是否超过 5MB 
        # Base64 编码后的体积大约是原图的 4/3，5MB ≈ 5 * 1024 * 1024 bytes
        # 加上 data:image/... 前缀，粗略按体积计算
        approx_size = len(img_b64) * 0.75
        if approx_size > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"Image at index {idx} exceeds the 5MB size limit.")

# ==========================================
# 4. 核心端点 /chat
# ==========================================
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, token: str = Depends(verify_authorization)):
    """
    唯一客服交互入口，兼容文本与图片咨询 
    仅仅支持 POST 请求 
    """
    # 1. 校验图片合法性
    if request.images:
        validate_images(request.images)
        
    # 2. 会话管理：若不传入，系统自动生成新 ID 
    current_session_id = request.session_id if request.session_id else f"kf_session_{uuid.uuid4().hex[:8]}"
    
    # 3. 组装载荷并调用你写好的 RAG 逻辑
    payload = {
        "question": request.question,
        "images": request.images,
        "session_id": current_session_id,
        "stream": request.stream
    }
    
    try:
        # 执行流水线，获取最终上下文状态
        ctx = pipeline.run(payload)
        
        # 4. 构建标准响应体 
        return ChatResponse(
            code=0,
            msg="success",
            data=ChatResponseData(
                answer=ctx.answer,  # 已经包含了 <PIC> 占位符的文本
                session_id=current_session_id,
                timestamp=int(time.time()), # 响应时间戳(秒) 
                returned_images=ctx.returned_images # 包含关联的图片 ID 列表
            )
        )
        
    except Exception as e:
        # 容错处理：确保即使内部抛错，也能返回合规的 JSON 结构
        return ChatResponse(
            code=500,
            msg=f"Internal Server Error: {str(e)}",
            data=ChatResponseData(
                answer="抱歉，系统处理您的请求时出现了内部异常。",
                session_id=current_session_id,
                timestamp=int(time.time())
            )
        )

if __name__ == "__main__":
    # 本地启动服务，默认端口 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
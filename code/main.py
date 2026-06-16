from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi.staticfiles import StaticFiles
import time
import uuid
import re
import os
import json

# 导入你在 answer.py 中写好的核心处理逻辑
from answer import generate_final_answer, generate_with_progress

# 初始化 FastAPI 实例
app = FastAPI(
    title="多模态客服智能体 API",
    description="适配客服比赛要求的标准 RESTful API 服务"
)

# 配置跨域资源共享 (CORS)，允许前端网页跨域调用本接口
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 比赛测试阶段允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ==========================================
# 环境变量与配置选项
# ==========================================
# 从文档提取的默认测试 Token：sk_customer_20260304
KAFU_API_TOKEN = os.getenv("KAFU_API_TOKEN", "sk_customer_20260304")

# 用于校验 Base64 图片格式的正则表达式（严格要求前缀格式和支持的后缀）
IMAGE_PREFIX_PATTERN = re.compile(r'^data:image/(png|jpg|jpeg|webp);base64,')


# ==========================================
# 数据模型定义 (严格对齐比赛文档)
# ==========================================

class ChatRequest(BaseModel):
    """请求体结构定义"""
    question: str = Field(..., min_length=1, description="用户的客服问题字符串")
    images: Optional[List[str]] = Field(default=[], description="Base64 格式图片列表，最多3张")
    session_id: Optional[str] = Field(default=None, description="客服会话 ID，用于多轮对话")
    stream: Optional[bool] = Field(default=False, description="是否流式响应，默认 FALSE")

class ChatResponseData(BaseModel):
    """响应体内的 data 结构"""
    answer: str
    session_id: str
    timestamp: int
    returned_images: List[str]  # 赛题要求：在数组中提供图片 ID

class ChatResponse(BaseModel):
    """完整响应体结构定义"""
    code: int = 0
    msg: str = "success"
    data: ChatResponseData


# ==========================================
# 依赖与校验逻辑
# ==========================================

async def verify_authorization(authorization: Optional[str] = Header(None)):
    """
    【认证规范校验】
    强制校验请求头中是否携带 Authorization: Bearer {KAFU_API_TOKEN}
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, 
            detail="Unauthorized: Missing or invalid Bearer token format."
        )
    
    # 提取并比对 Token
    token = authorization.split(" ")[1]
    if token != KAFU_API_TOKEN:
        raise HTTPException(
            status_code=401, 
            detail="Unauthorized: Invalid KAFU API Token."
        )
    
    return token

def validate_images(images: List[str]):
    """
    【多模态边界校验】
    1. 图片数量不能超过 3 张
    2. 图片必须严格符合 data:image/{png/jpg/jpeg/webp};base64,{编码内容} 格式
    3. 单张图片不能超过 5MB
    """
    if len(images) > 3:
        raise HTTPException(status_code=400, detail="Bad Request: 图片最多只支持上传 3 张。")
    
    for img in images:
        # 1. 严格校验赛题要求的 Base64 前缀
        if not IMAGE_PREFIX_PATTERN.match(img):
            raise HTTPException(
                status_code=400, 
                detail="Bad Request: 图片格式不合法，必须以 data:image/{png/jpg/jpeg/webp};base64, 作为前缀。"
            )

        # 2. 简单估算 Base64 体积 (5MB 原图转 Base64 长度大约为 5 * 1024 * 1024 * 1.33 ≈ 6.99MB)
        # 这里宽泛一点限制在 7.5 * 1024 * 1024 字符左右
        if len(img) > 7864320:
            raise HTTPException(status_code=400, detail="Bad Request: 单张图片大小不能超过 5MB。")


# ==========================================
# 核心接口路由
# ==========================================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, token: str = Depends(verify_authorization)):
    """
    核心多模态对话端点
    """
    try:
        # 1. 校验请求中的图片约束 (数量、格式、大小)
        validate_images(request.images)

        # 2. 会话 ID 处理：如果前端没有传，则后端自动生成一个新的 ID 并返回
        current_session_id = request.session_id if request.session_id else f"kf_session_{uuid.uuid4().hex[:8]}"

        print(f"\n[API 接收请求] Session: {current_session_id} | 包含图片数: {len(request.images)}")
        print(f"[API 接收问题] {request.question}")

        # 3. 核心调用：将问题和图片传入你的 RAG 生成函数
        if request.stream:
            # ====== SSE 流式路径 ======
            async def sse_event_stream():
                try:
                    for event in generate_with_progress(request.question, request.images, current_session_id):
                        event_type = event.get("type", "message")
                        payload = event.get("data", event)
                        yield f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    error_payload = json.dumps({"msg": f"处理请求时发生错误: {str(e)}"}, ensure_ascii=False)
                    yield f"event: error\ndata: {error_payload}\n\n"

            return StreamingResponse(
                sse_event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )

        # ====== 同步路径（原有逻辑不变） ======
        final_answer, extracted_image_ids = generate_final_answer(request.question, request.images, current_session_id)

        print(f"[API 生成回答] {final_answer}")
        print(f"[API 返回图片 ID 列表] {extracted_image_ids}")

        # 4. 组装并返回标准响应 JSON
        return {
            "code": 0,
            "msg": "success",
            "data": {
                "answer": final_answer,
                "session_id": current_session_id,
                "timestamp": int(time.time()),
                "returned_images": extracted_image_ids
            }
        }

    except HTTPException as he:
        # 将我们主动抛出的 400/401 校验异常直接返回给前端
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        # 兜底容错：即使发生未捕获异常，也返回符合格式的 JSON，防止服务崩溃
        return {
            "code": 500,
            "msg": f"Internal Server Error: {str(e)}",
            "data": {
                "answer": "非常抱歉，智能体在处理您的问题时遇到了内部错误，请稍后再试。",
                "session_id": request.session_id or f"kf_session_{uuid.uuid4().hex[:8]}",
                "timestamp": int(time.time()),
                "returned_images": []
            }
        }

app.mount("/images", StaticFiles(directory="/home/xjl/workspace/code/-agent/images"), name="images")

app.mount("/", StaticFiles(directory="/home/xjl/workspace/code/-agent/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # 启动服务器 (默认绑定 8000 端口，可通过 http://localhost:8000/docs 查看接口文档)
    print("🚀 正在启动多模态客服智能体 API 服务...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
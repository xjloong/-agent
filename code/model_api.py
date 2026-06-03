import os
from typing import List, Optional

from openai import APIConnectionError, OpenAI
import dashscope
from http import HTTPStatus

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_MODEL = "text-embedding-v4"
QWEN_MODEL = "qwen3.5-flash"
EMBED_DIM = 1024
API_TOKEN = "sk-c56434e9222e4c00a83997246eda6c85"  # 请替换为你自己的 API Key

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client

    if _client is not None:
        return _client

    # api_key = os.getenv("DASHSCOPE_API_KEY")
    api_key = API_TOKEN
    if not api_key:
        raise RuntimeError(
            "缺少环境变量 DASHSCOPE_API_KEY，无法调用 DashScope embedding 接口。"
        )

    _client = OpenAI(
        api_key=api_key,
        base_url=DASHSCOPE_BASE_URL,
    )
    return _client



def text_rerank(query: str, documents: List[str], top_n: int = 2) -> List[int]:
    """
    对文档进行重排序
    返回结果：重排后得分最高的文档在原列表中的索引顺序
    """
    resp = dashscope.TextReRank.call(
        model="qwen3-rerank",
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=False, # 设置为 False，我们只需要知道哪些文档排在前面
        instruct="给定产品手册查询，检索出能够准确回答问题的段落。"
    )
    
    if resp.status_code == HTTPStatus.OK:
        # 提取重排后的结果索引
        # resp.output.results 包含了按得分排序后的结果，其中的 index 对应原 documents 的下标
        return [result.index for result in resp.output.results]
    else:
        print(f"Rerank 失败: {resp.code}, {resp.message}")
        return []


def compute_embedding(text: str) -> List[float]:
    try:
        completion = get_client().embeddings.create(
            model=EMBED_MODEL,
            input=text,
            dimensions=EMBED_DIM,
            encoding_format="float",
        )
    except APIConnectionError as exc:
        raise RuntimeError(
            "无法连接text-embedding-v4接口。请检查网络连通性、代理配置，"
            f"以及服务地址 {DASHSCOPE_BASE_URL} 是否可访问。"
        ) from exc

    return completion.data[0].embedding

def ask_qwen(system_prompt: str, user_text: str) -> str:
    try:
        completion = get_client().chat.completions.create(
            model=QWEN_MODEL,
            messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_text}
        ]
        )
    except APIConnectionError as exc:
        raise RuntimeError(
            "无法连接 Qwen3.5-Flash 接口。请检查网络连通性、代理配置，"
            f"以及服务地址 {DASHSCOPE_BASE_URL} 是否可访问。"
        ) from exc

    return completion.choices[0].message.content

def ask_images(images: List[str], question: str) -> str:
    """
    调用 Qwen 多模态大模型解析 Base64 图片，提取关键意图
    """
    content = []
    
    # 赛题约定的格式是 data:image/{png/jpg...};base64,...
    # Qwen 接口完美兼容这种带 data URI 协议头的 Base64 字符串
    for img_base64 in images:
        content.append({"image": img_base64})
        
    # 构造针对性的视觉 Prompt
    vision_prompt = (
        f"用户上传了相关客服凭证图片，并提出了问题：【{question}】。\n"
        "请结合图片，提取出与问题相关的核心要素（如产品型号、指示灯状态、错误代码、物流状态等），"
        "并用简短的文字概括这些信息，以便后续客服系统理解用户意图。"
    )
    content.append({"text": vision_prompt})
    
    messages = [{"role": "user", "content": content}]
    
    try:
        # 注意：建议使用专门的多模态模型（如 qwen3.7-plus 或 qwen-vl-max）
        # 确保环境变量 DASHSCOPE_API_KEY 已配置
        response = dashscope.MultiModalConversation.call(
            api_key=API_TOKEN,
            model='qwen-vl-max', 
            messages=messages
        )
        # 提取解析出的文本描述
        return response.output.choices[0].message.content[0]["text"]
    except Exception as e:
        print(f"多模态大模型解析异常: {e}")
        return "" # 降级处理：若解析失败，返回空字符串，不阻塞主流程

if __name__ == "__main__":
    # print(compute_embedding("你好"))
    # print(ask_qwen("回答助手","你是谁？"))
    print(text_rerank("如何更换电池？", ["更换电池的方法是...", "这个问题没有提到电池", "请按照以下步骤更换电池..."], top_n=2))

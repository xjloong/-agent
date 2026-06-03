import json
import os
from typing import List, Dict
from dataclasses import dataclass, field
from pymilvus import MilvusClient
from model_api import compute_embedding, ask_qwen, text_rerank,ask_images
from pprint import pprint as pp
import base64
import mimetypes
import os

MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"
COLLECTION_NAME = "product_manuals"

client = MilvusClient(
    uri=MILVUS_URI,
    token=MILVUS_TOKEN,
)

# ==========================================
# 1. 定义贯穿生命周期的数据对象
# ==========================================
@dataclass
class RAGContext:
    question: str
    images: List[str] = field(default_factory=list)
    session_id: str = ""
    stream: str = "false"
    
    # 内部流转状态
    target_doc: str = "None"
    refined_query: str = "None"
    search_txt: List[Dict] = field(default_factory=list)
    answer: str = "None"
    returned_images: List[str] = field(default_factory=list)

def encode_image_to_base64(image_path: str) -> str:
    """
    将本地图片转换为赛题 API 要求的带前缀的 Base64 格式。
    赛题规范: data:image/{ext};base64,{base64_str}
    """
    if not os.path.exists(image_path):
        print(f"⚠️ 警告: 找不到测试图片路径: {image_path}")
        return ""
        
    # 自动识别图片的 MIME 类型 (如 image/png, image/jpeg)
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith('image/'):
        mime_type = 'image/jpeg' # 默认容错兜底
        
    try:
        with open(image_path, "rb") as image_file:
            # 读取二进制并进行 Base64 编码
            encoded_bytes = base64.b64encode(image_file.read())
            encoded_string = encoded_bytes.decode('utf-8')
            
        # 严格按照赛事规范拼接前缀
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"⚠️ 图片编码失败: {e}")
        return ""

def load_doc_names(json_path: str = "doc_names.json") -> List[str]:
    """读取包含所有产品手册名称的 JSON 文件"""
    if not os.path.exists(json_path):
        print(f"警告: 找不到 {json_path} 文件。请先运行建库脚本。")
        return []
    
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_json_response(raw_text: str) -> str:
    """清理大模型输出中可能包含的 markdown 符号，确保 JSON 可解析"""
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# ==========================================
# 2. 核心流水线类
# ==========================================
class RAGPipeline:
    def __init__(self):
        self.doc_names = load_doc_names()
        
    def step1_process_images(self, ctx: RAGContext):
        """步骤 1：处理图片"""
        if ctx.images and len(ctx.images) > 0:
            print("=> [Step 1] 检测到图片，正在融合图文信息...")
            ctx.question = ask_images(ctx.images, ctx.question)
        else:
            print("=> [Step 1] 无图片输入，继续处理纯文本。")

    def step2_route_and_refine(self, ctx: RAGContext):
        """步骤 2：一步完成意图识别与查询优化 (或直接回答)"""
        print("=> [Step 2] 正在判断意图并优化查询...")
        
        # 精心设计的结构化 Prompt，强制模型输出 JSON
        system_prompt = f"""
你是一个专业的智能客服路由与解答助手。
当前可用的产品手册列表如下：
{json.dumps(self.doc_names, ensure_ascii=False)}

【任务指令】
请分析用户的问题，判断是否需要检索具体的产品手册，并严格按照以下 JSON 格式输出结果，不要包含任何多余的解释性文字。

【情况 A：不需要查阅手册】
如果用户是在闲聊、问候（如“你好”）、抱怨物流（如“待揽收什么原因”）、或者问题明显超出产品技术支持范围，请直接作答。
输出格式：
{{
    "action": "direct_answer",
    "answer": "你给出的详细、礼貌的回答"
}}

【情况 B：需要查阅特定产品手册】
如果用户询问具体的使用方法、故障、零件等，且属于上述手册列表内的产品。
输出格式：
{{
    "action": "search",
    "target_doc": "准确的手册名称(必须是列表中的一个)",
    "refined_query": "提取核心搜索短语(去除礼貌用语，不包含产品名本身)"
}}
"""
        raw_response = ask_qwen(system_prompt=system_prompt, user_text=ctx.question)
        
        try:
            # 解析大模型返回的 JSON
            json_str = clean_json_response(raw_response)
            result = json.loads(json_str)
            
            if result.get("action") == "direct_answer":
                ctx.answer = result.get("answer", "我暂时无法回答这个问题。")
                print("=> [Step 2 结论] 无需检索，直接生成回答。")
            else:
                ctx.target_doc = result.get("target_doc", "None")
                ctx.refined_query = result.get("refined_query", "None")
                print(f"=> [Step 2 结论] 需要检索。目标手册: {ctx.target_doc}, 优化查询: {ctx.refined_query}")
                
        except json.JSONDecodeError:
            print(f"解析路由大模型输出失败。原始输出: {raw_response}")
            ctx.answer = "抱歉，系统在理解您的问题时出现了内部解析错误，请换个说法重试。"

    def step3_search_and_generate(self, ctx: RAGContext):
        """步骤 3：执行检索并生成最终答案"""
        if ctx.answer != "None":
            return
            
        if ctx.target_doc not in self.doc_names:
            ctx.answer = "抱歉，我未能找到与您问题相匹配的产品手册库。"
            return

        print(f"=> [Step 3] 正在 Milvus 中检索 {ctx.target_doc} ...")
        
        query_vector = compute_embedding(ctx.refined_query)
        
        # 确保 output_fields 中包含了 "image" 字段
        search_res = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            filter=f'doc_name == "{ctx.target_doc}"',
            output_fields=["doc_name", "content", "image"], 
            limit=3,
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}}
        )
        
        # 将检索结果组装成字典列表
        if search_res and len(search_res[0]) > 0:
            for hits in search_res[0]:
                entity = hits.get("entity")
                content_text = entity.get("content", "")
                
                # 兼容处理：确保 image 字段是列表类型
                image_data = entity.get("image", [])
                if isinstance(image_data, str):
                    try:
                        image_data = json.loads(image_data)
                    except json.JSONDecodeError:
                        image_data = [image_data] if image_data else []
                
                ctx.search_txt.append({
                    "content": content_text,
                    "images": image_data
                })
                # 将涉及到的图片 ID 加入到最终返回列表中
                ctx.returned_images.extend(image_data)
                
        if not ctx.search_txt:
            ctx.answer = "抱歉，在手册中没有找到与您问题相关的确切信息。"
            return
            
        # 为大模型构建结构化的参考上下文
        formatted_context = ""
        for idx, item in enumerate(ctx.search_txt):
            formatted_context += f"【参考段落 {idx+1}】\n内容：{item['content']}\n关联图片ID：{item['images']}\n---\n"
        
        print("=> [Step 3] 检索完成，正在生成最终回答...")
        
        # 强化 Prompt，要求大模型根据参考段落中的“关联图片ID”输出 <PIC> 占位符
        qa_system_prompt = """
你是一个专业的产品技术支持专家。
请严格基于下面提供的【参考手册内容】，回答用户的【问题】。

【输出要求】：
1. 必须完全基于参考内容，不能编造。未提及的信息请直接说明未找到。
2. 回答需条理清晰，操作步骤请使用列表排版。
3. 如果参考内容中包含【关联图片ID】，请务必在你回答的相关文字位置插入严格等于该图片数量的 `<PIC>` 标识。例如，如果该步骤对应2个图片ID，请输出“...完成操作。<PIC><PIC>”。
"""
        user_input_prompt = f"【参考手册内容】:\n{formatted_context}\n\n【用户问题】:\n{ctx.question}"
        
        ctx.answer = ask_qwen(system_prompt=qa_system_prompt, user_text=user_input_prompt)

    def run(self, request_payload: Dict) -> RAGContext:
        """运行完整流水线"""
        # 初始化状态对象
        ctx = RAGContext(
            question=request_payload.get("question", ""),
            images=request_payload.get("images", []),
            session_id=request_payload.get("session_id", ""),
            stream=str(request_payload.get("stream", "false")).lower()
        )
        
        # 依次执行管线
        self.step1_process_images(ctx)
        self.step2_route_and_refine(ctx)
        self.step3_search_and_generate(ctx)
        
        return ctx
    
if __name__ == "__main__":

    test_payload_1 = {
        "question": "运动手表如何更换表带",
        "images": [],
        "session_id": "12345",
        "stream": "false"
    }
    
    test_image_path = r"D:\Code\-agent\data\KownledgeBase\手册\插图\Blower_01.png"
    base64_image_data = encode_image_to_base64(test_image_path)
    
    test_payload_2 = {
        "question": "图片说了什么",
        # 如果成功转码，则放入列表，否则传空列表避免接口崩溃
        "images": [base64_image_data] if base64_image_data else [], 
        "session_id": "12345",
        "stream": "false"
    }

    pipeline = RAGPipeline()
    
    final_ctx_1 = pipeline.run(test_payload_1)
    print(f"最终输出对象属性:\n{final_ctx_1}")
   
    final_ctx_2 = pipeline.run(test_payload_2)
    print(f"最终输出对象属性:\n{final_ctx_2}")
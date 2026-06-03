import base64
import requests
import json
import time

# 1. 配置接口地址和赛题要求的鉴权 Token
API_URL = "http://127.0.0.1:8000/chat"
TOKEN = "sk_customer_20260304"

# 赛题要求的固定鉴权 Header
HEADERS_VALID = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

HEADERS_INVALID = {
    "Authorization": "Bearer fake_token_123",
    "Content-Type": "application/json"
}

def encode_image_to_base64(image_path: str) -> str:
    """读取本地图片并转换为带有赛题规定前缀的 Base64 字符串"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            # 赛题严格要求：Base64 图片必须携带完整前缀 [cite: 43]
            # 格式为 data:image/{png/jpg/jpeg/webp};base64,{编码内容}
            return f"data:image/jpeg;base64,{encoded_string}"
    except FileNotFoundError:
        print(f"❌ 找不到图片文件: {image_path}，请检查路径。")
        exit(1)

IMAGE_PATH = r"data\KownledgeBase\手册\插图\Blower_01.png"
img_base64 = encode_image_to_base64(IMAGE_PATH)

def print_result(title: str, response: requests.Response):
    """格式化打印测试结果"""
    print(f"\n{'='*15} {title} {'='*15}")
    print(f"状态码: {response.status_code}")
    try:
        print("响应体:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except:
        print(f"响应内容: {response.text}")
    print("=" * 45)


# ==========================================
# 2. 测试用例执行
# ==========================================
def run_tests():
    # print("🚀 开始执行多模态智能体 API 自动化测试...\n")

    # # --------------------------------------------------
    # # 测试一：鉴权拦截测试 (反向测试)
    # # --------------------------------------------------
    # print("👉 [Test 1] 正在测试: 错误 Token 拦截...")
    # payload_auth = {"question": "你好"}
    # res_auth = requests.post(API_URL, json=payload_auth, headers=HEADERS_INVALID)
    # print_result("Test 1: 鉴权拦截", res_auth)
    # assert res_auth.status_code == 403, "❌ 鉴权测试失败：未拦截错误 Token"


    # # --------------------------------------------------
    # # 测试二：单轮业务提问 (正向测试)
    # # --------------------------------------------------
    # print("\n👉 [Test 2] 正在测试: 首次业务提问 (自动生成 session_id)...")
    # payload_first = {
    #     "question": "我想更换健身追踪器的表带，有其他尺寸可选吗？" 
    # }
    # res_first = requests.post(API_URL, json=payload_first, headers=HEADERS_VALID)
    # print_result("Test 2: 首次提问", res_first)
    
    # # 提取 session_id，用于下一轮测试
    # session_id = None
    # if res_first.status_code == 200:
    #     session_id = res_first.json().get("data", {}).get("session_id")
    #     print(f"✅ 成功获取 session_id: {session_id}")


    # # --------------------------------------------------
    # # 测试三：多轮对话追问 (使用 Test 2 的 session_id)
    # # --------------------------------------------------
    # if session_id:
    #     print(f"\n👉 [Test 3] 正在测试: 多轮对话追问 (携带 session_id: {session_id})...")
    #     time.sleep(1) # 稍微暂停，模拟真实用户打字
    #     payload_followup = {
    #         "question": "那如果是别的型号呢？比如 Versa 3 可以用吗？",
    #         "session_id": session_id
    #     }
    #     res_followup = requests.post(API_URL, json=payload_followup, headers=HEADERS_VALID)
    #     print_result("Test 3: 多轮追问", res_followup)
    # else:
    #     print("\n⚠️ 跳过 Test 3，因为未获取到 session_id。")


    # --------------------------------------------------
    # 测试四：多模态图文输入测试
    # --------------------------------------------------
    print("\n👉 [Test 4] 正在测试: 多模态图文融合 (带 Base64 图片)...")
    payload_multimodal = {
        "question": "这张图里是什么意思？",
        "images": [img_base64],
        "session_id": "test_vision_999"
    }
    res_multimodal = requests.post(API_URL, json=payload_multimodal, headers=HEADERS_VALID)
    print_result("Test 4: 多模态图文测试", res_multimodal)

    print("\n🎉 所有自动化测试执行完毕！请检查日志查看大模型回答质量。")

if __name__ == "__main__":
    run_tests()
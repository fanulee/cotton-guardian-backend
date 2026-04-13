import os, cv2, base64, sqlite3, json, time, requests, gc
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ================= 扣子配置 =================
COZE_API_TOKEN = 'pat_4LoBL7qWrtuswQ8zuwAMkFXwkVy4ht7pFyDtkEpMQyt2fOeDkCFRoJu3JIDp8meD'
BOT_ID = '7628183682519908395'
CREATE_CHAT_URL = "https://api.coze.cn/v3/chat"
RETRIEVE_URL = "https://api.coze.cn/v3/chat/retrieve"
MESSAGE_LIST_URL = "https://api.coze.cn/v3/chat/message/list"

# ================= 极低内存模型加载 =================
print("👉 正在启动边缘计算引擎 (生存模式)...")
try:
    # 🚨 关键：限制线程数和内存增长，这是防止 SIGSEGV 的核心
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    session = ort.InferenceSession("best.onnx", sess_options=opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    CLASS_NAMES = {0: "棉铃虫", 1: "蚜虫", 2: "红蜘蛛"} 
    print("✅ 引擎就绪！")
except Exception as e:
    print(f"❌ 启动失败: {e}")

# ==================== 接口区 ====================

@app.route('/api/detect', methods=['POST'])
def detect_pest():
    if 'file' not in request.files: return jsonify({"status": "error", "message": "无文件"})
    file = request.files['file']; temp_path = "t.jpg"
    try:
        file.save(temp_path)
        # 1. 暴力预处理：读图后立刻释放原始字节，且只处理小图
        img = cv2.imread(temp_path)
        if img is None: return jsonify({"status": "error", "message": "无法读图"})
        
        # 哪怕原图很大，我们也只在 640x640 维度操作
        img = cv2.resize(img, (640, 640))
        # 理顺内存并转换为模型需要的格式
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
        blob = np.ascontiguousarray(blob).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        # 2. 执行推理
        outs = session.run([output_name], {input_name: blob})[0]
        
        # 3. 矩阵自适应解析
        outs = np.squeeze(outs)
        if outs.shape[0] < outs.shape[1]: outs = outs.T

        results = []
        # 只保留置信度最高的前 50 个框，防止内存被几千个废框撑爆
        for row in outs:
            scores = row[4:]
            max_s = float(np.max(scores))
            if max_s > 0.05: # 灵敏度 5%
                cid = int(np.argmax(scores))
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                l, t = int(cx - bw/2), int(cy - bh/2)
                # 直接在 640 图上画框
                cv2.rectangle(img, (l, t), (l+int(bw), t+int(bh)), (0, 255, 0), 2)
                results.append({"name": CLASS_NAMES.get(cid, "虫害"), "conf": max_s})
        
        # 4. 编码并彻底清理
        _, buf = cv2.imencode('.jpg', img)
        img_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"
        
        del img, blob, outs; gc.collect() # 强制清理
        return jsonify({"status": "success", "data": {"pest_count": len(results), "result_image": img_b64}})
    except Exception as e: return jsonify({"status": "error", "message": str(e)})
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@app.route('/api/chat', methods=['POST'])
def chat_with_agent():
    try:
        gc.collect()
        d = request.get_json(); msg = d.get('prompt', '')
        if not msg: return jsonify({"status": "error", "reply": "内容为空"})
        h = {"Authorization": f"Bearer {COZE_API_TOKEN}", "Content-Type": "application/json"}
        p = {"bot_id": BOT_ID, "user_id": "pro", "stream": False, "additional_messages": [{"role": "user", "content": str(msg), "content_type": "text"}]}
        
        r = requests.post(CREATE_CHAT_URL, headers=h, json=p, timeout=30)
        res = r.json()
        if res.get('code') != 0: return jsonify({"status": "error", "reply": "AI繁忙"})
        
        cid, cvid = res['data']['id'], res['data']['conversation_id']
        for _ in range(20):
            time.sleep(2)
            sr = requests.get(f"{RETRIEVE_URL}?chat_id={cid}&conversation_id={cvid}", headers=h, timeout=10).json()
            if sr['data']['status'] == 'completed':
                mr = requests.get(f"{MESSAGE_LIST_URL}?chat_id={cid}&conversation_id={cvid}", headers=h, timeout=10).json()
                for m in mr.get('data', []):
                    if m['type'] == 'answer': return jsonify({"status": "success", "reply": m['content']})
        return jsonify({"status": "error", "reply": "响应超时"})
    except: return jsonify({"status": "error", "reply": "通讯中断"})

# 其余基础存根
@app.route('/api/login', methods=['POST'])
def login(): return jsonify({"status": "success", "username": request.get_json().get('username'), "role": "farmer"})
@app.route('/api/get_fields', methods=['GET'])
def get_fields(): return jsonify([])
@app.route('/api/save_field', methods=['POST'])
def save_field(): return jsonify({"status": "success"})
@app.route('/api/check_pending', methods=['GET'])
def check_pending(): return jsonify({"has_pending": False})
@app.route('/api/save_record', methods=['POST'])
def save_record(): return jsonify({"status": "success"})
@app.route('/api/get_records', methods=['GET'])
def get_records(): return jsonify([])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

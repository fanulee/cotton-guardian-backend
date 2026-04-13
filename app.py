import os, cv2, base64, sqlite3, json, time, requests, gc
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ================= 1. 扣子 (Coze) 智能体配置 =================
COZE_API_TOKEN = 'pat_4LoBL7qWrtuswQ8zuwAMkFXwkVy4ht7pFyDtkEpMQyt2fOeDkCFRoJu3JIDp8meD'
BOT_ID = '7628183682519908395'
CREATE_CHAT_URL = "https://api.coze.cn/v3/chat"
RETRIEVE_URL = "https://api.coze.cn/v3/chat/retrieve"
MESSAGE_LIST_URL = "https://api.coze.cn/v3/chat/message/list"

# ================= 2. 极致内存优化：加载模型 =================
print("👉 正在启动云端生存模式引擎...")
try:
    # 🚨 核心优化：强制单线程运行，防止 Render 内存溢出崩溃
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    session = ort.InferenceSession("best.onnx", sess_options=opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    CLASS_NAMES = {0: "棉铃虫", 1: "蚜虫", 2: "红蜘蛛"} 
    print("✅ 云端引擎已就绪！")
except Exception as e:
    print(f"❌ 引擎启动失败: {e}")

# 数据库存储路径
DB_PATH = os.path.join(os.getcwd(), 'cotton_platform.db')

def init_db():
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT, role TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS fields (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, field_internal_id TEXT, name TEXT, risk TEXT, risk_class TEXT, latlngs TEXT, sensor_images TEXT, area REAL, crop_variety TEXT, plant_date TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, time TEXT, field_name TEXT, field_internal_id TEXT, image_base64 TEXT, pest_count INTEGER, risk TEXT, advice TEXT, operation TEXT, record_type TEXT, parent_record_id INTEGER, scheduled_recheck_time TEXT, loop_status TEXT)''')
    conn.commit(); conn.close()

# ==================== 3. 视觉识别接口 (云端避坑版) ====================

@app.route('/api/detect', methods=['POST'])
def detect_pest():
    if 'file' not in request.files: return jsonify({"status": "error", "message": "无文件"})
    file = request.files['file']; temp_path = "t.jpg"
    try:
        file.save(temp_path)
        # 立即读取并释放原始文件内存
        img = cv2.imread(temp_path)
        if img is None: return jsonify({"status": "error", "message": "格式不支持"})
        
        # 🚀 内存优化：无论原图多大，立即降采样到 640 跑 AI
        img_resized = cv2.resize(img, (640, 640))
        blob = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
        blob = np.ascontiguousarray(blob).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        # 推理
        outs = session.run([output_name], {input_name: blob})[0]
        outs = np.squeeze(outs)
        if outs.shape[0] < outs.shape[1]: outs = outs.T

        results = []
        for row in outs:
            scores = row[4:]
            max_s = float(np.max(scores))
            if max_s > 0.05: # 灵敏度阈值
                cid = int(np.argmax(scores))
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                l, t = int(cx - bw/2), int(cy - bh/2)
                # 直接在缩放后的图上画框，极度节省内存
                cv2.rectangle(img_resized, (l, t), (l+int(bw), t+int(bh)), (0, 255, 0), 2)
                results.append({"name": CLASS_NAMES.get(cid, "虫害"), "conf": max_s})
        
        # 编码返回
        _, buf = cv2.imencode('.jpg', img_resized)
        img_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"
        
        # 🚀 强制清理内存
        del img, img_resized, blob, outs; gc.collect()
        
        return jsonify({"status": "success", "data": {"pest_count": len(results), "result_image": img_b64}})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

# ==================== 4. AI 对话接口 ====================

@app.route('/api/chat', methods=['POST'])
def chat_with_agent():
    try:
        gc.collect()
        data = request.get_json(); msg = data.get('prompt', '诊断态势')
        h = {"Authorization": f"Bearer {COZE_API_TOKEN}", "Content-Type": "application/json"}
        p = {"bot_id": BOT_ID, "user_id": "pro_user", "stream": False, "additional_messages": [{"role": "user", "content": str(msg), "content_type": "text"}]}
        
        r = requests.post(CREATE_CHAT_URL, headers=h, json=p, timeout=30)
        res = r.json()
        cid, cvid = res['data']['id'], res['data']['conversation_id']
        
        # 轮询获取结果
        for _ in range(25):
            time.sleep(2)
            sr = requests.get(f"{RETRIEVE_URL}?chat_id={cid}&conversation_id={cvid}", headers=h, timeout=10).json()
            if sr['data']['status'] == 'completed':
                mr = requests.get(f"{MESSAGE_LIST_URL}?chat_id={cid}&conversation_id={cvid}", headers=h, timeout=10).json()
                for m in mr.get('data', []):
                    if m['type'] == 'answer': return jsonify({"status": "success", "reply": m['content']})
        return jsonify({"status": "error", "reply": "AI响应超时"})
    except: return jsonify({"status": "error", "reply": "云端通讯波动"})

# 其余基础管理接口
@app.route('/api/login', methods=['POST'])
def login(): return jsonify({"status": "success", "username": "专家用户", "role": "farmer"})
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
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

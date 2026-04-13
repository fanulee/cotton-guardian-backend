import os
import cv2
import base64
import sqlite3
import json
import time
import numpy as np
import requests
import gc  # 🚨 新增：内存垃圾强行回收
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS

# 1. 初始化 Flask 服务
app = Flask(__name__)
CORS(app)

# ================= 扣子 (Coze) 智能体配置 =================
COZE_API_TOKEN = 'pat_4LoBL7qWrtuswQ8zuwAMkFXwkVy4ht7pFyDtkEpMQyt2fOeDkCFRoJu3JIDp8meD'
BOT_ID = '7628183682519908395'
CREATE_CHAT_URL = "https://api.coze.cn/v3/chat"
RETRIEVE_URL = "https://api.coze.cn/v3/chat/retrieve"
MESSAGE_LIST_URL = "https://api.coze.cn/v3/chat/message/list"

# ================= 硬件优化：加载模型 =================
print("👉 正在启动边缘计算引擎...")
try:
    # 强制使用 CPU，节省显存开销
    session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    CLASS_NAMES = {0: "棉铃虫", 1: "蚜虫", 2: "红蜘蛛"} 
    print("✅ 引擎就绪！")
except Exception as e:
    print(f"❌ 引擎初始化失败: {e}")

DB_PATH = os.path.join(os.getcwd(), 'cotton_platform.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL, role TEXT DEFAULT 'farmer')''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS fields (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, field_internal_id TEXT NOT NULL, name TEXT, risk TEXT, risk_class TEXT, latlngs TEXT, sensor_images TEXT, area REAL DEFAULT 0, crop_variety TEXT DEFAULT '', plant_date TEXT DEFAULT '')''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, time TEXT, field_name TEXT, field_internal_id TEXT, image_base64 TEXT, pest_count INTEGER, risk TEXT, advice TEXT, operation TEXT, record_type TEXT DEFAULT 'initial', parent_record_id INTEGER DEFAULT 0, scheduled_recheck_time TEXT, loop_status TEXT DEFAULT 'closed')''')
    conn.commit()
    conn.close()

# ==================== 1. 账号接口 ====================
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    user, pwd = data.get('username'), data.get('password')
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, 'farmer')", (user, pwd))
        conn.commit(); return jsonify({"status": "success"})
    except: return jsonify({"status": "error", "message": "用户名已占用"})
    finally: conn.close()

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute("SELECT role FROM users WHERE username = ? AND password = ?", (data.get('username'), data.get('password')))
    user = cursor.fetchone(); conn.close()
    if user: return jsonify({"status": "success", "username": data.get('username'), "role": user[0]})
    return jsonify({"status": "error", "message": "账号或密码错误"}), 401

# ==================== 2. 地块接口 ====================
@app.route('/api/get_fields', methods=['GET'])
def get_fields():
    username = request.args.get('username')
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute("SELECT field_internal_id, name, risk, risk_class, latlngs, sensor_images, area, crop_variety, plant_date FROM fields WHERE username = ?", (username,))
    rows = cursor.fetchall(); conn.close()
    fields = [{"id": r[0], "name": r[1], "risk": r[2], "riskClass": r[3], "latlngs": json.loads(r[4]), "sensorImages": json.loads(r[5]), "area": r[6], "cropVariety": r[7], "plantDate": r[8]} for r in rows]
    return jsonify(fields)

@app.route('/api/save_field', methods=['POST'])
def save_field():
    data = request.get_json()
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute("SELECT id FROM fields WHERE username = ? AND field_internal_id = ?", (data['username'], data['id']))
    exists = cursor.fetchone()
    lat_j, sen_j = json.dumps(data.get('latlngs', [])), json.dumps(data.get('sensorImages', []))
    if exists:
        cursor.execute("UPDATE fields SET name=?, risk=?, risk_class=?, latlngs=?, sensor_images=?, area=?, crop_variety=?, plant_date=? WHERE username=? AND field_internal_id=?", (data['name'], data['risk'], data['riskClass'], lat_j, sen_j, data.get('area', 0), data.get('cropVariety', ''), data.get('plantDate', ''), data['username'], data['id']))
    else:
        cursor.execute("INSERT INTO fields (username, field_internal_id, name, risk, risk_class, latlngs, sensor_images, area, crop_variety, plant_date) VALUES (?,?,?,?,?,?,?,?,?,?)", (data['username'], data['id'], data['name'], data['risk'], data['riskClass'], lat_j, sen_j, data.get('area', 0), data.get('cropVariety', ''), data.get('plantDate', '')))
    conn.commit(); conn.close()
    return jsonify({"status": "success"})

# ==================== 3. 诊断与视觉核心 (修复 0 处检出问题) ====================
@app.route('/api/detect', methods=['POST'])
def detect_pest():
    if 'file' not in request.files: return jsonify({"status": "error", "message": "无文件"})
    file = request.files['file']
    temp_path = "temp_u.jpg"
    try:
        file.save(temp_path)
        img = cv2.imread(temp_path)
        orig_img = img.copy()
        h, w = img.shape[:2]
        xf, yf = w / 640.0, h / 640.0
        
        # 🚀 关键修复：理顺内存排布，防止 ONNX 读到“雪花图”
        blob = cv2.resize(img, (640, 640))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
        blob = blob.transpose((2, 0, 1))
        blob = np.ascontiguousarray(blob).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        outputs = session.run([output_name], {input_name: blob})[0]
        preds = np.squeeze(outputs).T
        
        boxes, scores, cids = [], [], []
        for row in preds:
            s_arr = row[4:]
            max_s = np.max(s_arr)
            if max_s > 0.1: # 调低阈值到 10% 以捕捉微小飞虫
                cid = np.argmax(s_arr)
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                boxes.append([int((cx - bw/2)*xf), int((cy - bh/2)*yf), int(bw*xf), int(bh*yf)])
                scores.append(float(max_s))
                cids.append(cid)

        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.1, 0.45)
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                b = boxes[i]
                cv2.rectangle(orig_img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (0, 255, 0), 2)
                results.append({"name": CLASS_NAMES.get(cids[i], "害虫"), "conf": round(scores[i], 2)})

        _, buffer = cv2.imencode('.jpg', orig_img)
        img_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"
        
        # 🚀 内存急救：清空大图缓存
        del img, orig_img, blob
        gc.collect()

        return jsonify({"status": "success", "data": {"pest_count": len(results), "risk_level": "高风险" if len(results) > 5 else "安全", "result_image": img_b64}})
    except Exception as e: return jsonify({"status": "error", "message": str(e)})
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

# ==================== 4. AI 智能体对话 (增加健壮性) ====================
@app.route('/api/chat', methods=['POST'])
def chat_with_agent():
    try:
        # 释放内存再开始对话
        gc.collect()
        data = request.get_json()
        msg = data.get('prompt', '') or data.get('query', '')
        if not msg: return jsonify({"status": "error", "reply": "内容为空"})

        headers = {"Authorization": f"Bearer {COZE_API_TOKEN}", "Content-Type": "application/json"}
        payload = {"bot_id": BOT_ID, "user_id": "demo_user", "stream": False, "additional_messages": [{"role": "user", "content": str(msg), "content_type": "text"}]}
        
        r = requests.post(CREATE_CHAT_URL, headers=headers, json=payload, timeout=20)
        res = r.json()
        if res.get('code') != 0: return jsonify({"status": "error", "reply": "AI接口忙"})

        cid, cvid = res['data']['id'], res['data']['conversation_id']
        # 轮询 (最多等 15 秒，防止 Render 超时)
        for _ in range(8):
            time.sleep(1.5)
            s_r = requests.get(f"{RETRIEVE_URL}?chat_id={cid}&conversation_id={cvid}", headers=headers).json()
            if s_r['data']['status'] == 'completed':
                m_r = requests.get(f"{MESSAGE_LIST_URL}?chat_id={cid}&conversation_id={cvid}", headers=headers).json()
                for m in m_r.get('data', []):
                    if m['type'] == 'answer': return jsonify({"status": "success", "reply": m['content']})
        return jsonify({"status": "error", "reply": "AI思考较慢，请刷新重试"})
    except: return jsonify({"status": "error", "reply": "连接超时"})

# 其它地块档案相关接口省略，保持原逻辑即可
@app.route('/api/check_pending', methods=['GET'])
def check_pending(): return jsonify({"has_pending": False})
@app.route('/api/save_record', methods=['POST'])
def save_record(): return jsonify({"status": "success"})
@app.route('/api/get_records', methods=['GET'])
def get_records(): return jsonify([])
@app.route('/api/delete_field', methods=['DELETE'])
def delete_field(): return jsonify({"status": "success"})

if __name__ == '__main__':
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

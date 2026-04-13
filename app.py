import os, cv2, base64, sqlite3, json, time, requests, gc
import numpy as np
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
    # 强制使用 CPU 推理，避免 Render 免费显存溢出
    session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    CLASS_NAMES = {0: "棉铃虫", 1: "蚜虫", 2: "红蜘蛛"} 
    print("✅ 引擎就绪！")
except Exception as e:
    print(f"❌ 引擎初始化失败: {e}")

DB_PATH = os.path.join(os.getcwd(), 'cotton_platform.db')

def init_db():
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL, role TEXT DEFAULT 'farmer')''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS fields (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, field_internal_id TEXT NOT NULL, name TEXT, risk TEXT, risk_class TEXT, latlngs TEXT, sensor_images TEXT, area REAL DEFAULT 0, crop_variety TEXT DEFAULT '', plant_date TEXT DEFAULT '')''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, time TEXT, field_name TEXT, field_internal_id TEXT, image_base64 TEXT, pest_count INTEGER, risk TEXT, advice TEXT, operation TEXT, record_type TEXT DEFAULT 'initial', parent_record_id INTEGER DEFAULT 0, scheduled_recheck_time TEXT, loop_status TEXT DEFAULT 'closed')''')
    conn.commit(); conn.close()

# ==================== 1. 账号与鉴权接口 ====================
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    user, pwd = data.get('username'), data.get('password')
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, 'farmer')", (user, pwd))
        conn.commit(); return jsonify({"status": "success"})
    except: return jsonify({"status": "error", "message": "用户名已被占用"})
    finally: conn.close()

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute("SELECT role FROM users WHERE username = ? AND password = ?", (data.get('username'), data.get('password')))
    user = cursor.fetchone(); conn.close()
    if user: return jsonify({"status": "success", "username": data.get('username'), "role": user[0]})
    return jsonify({"status": "error", "message": "账号或密码错误"}), 401

# ==================== 2. 地块与档案管理接口 ====================
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
    d = request.get_json(); conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    lat_j, sen_j = json.dumps(d.get('latlngs', [])), json.dumps(d.get('sensorImages', []))
    cursor.execute("INSERT OR REPLACE INTO fields (username, field_internal_id, name, risk, risk_class, latlngs, sensor_images, area, crop_variety, plant_date) VALUES (?,?,?,?,?,?,?,?,?,?)", (d['username'], d['id'], d['name'], d['risk'], d['riskClass'], lat_j, sen_j, d.get('area', 0), d.get('cropVariety', ''), d.get('plantDate', '')))
    conn.commit(); conn.close(); return jsonify({"status": "success"})

@app.route('/api/delete_field', methods=['DELETE'])
def delete_field():
    username = request.args.get('username'); field_id = request.args.get('id')
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute("DELETE FROM fields WHERE username = ? AND field_internal_id = ?", (username, field_id))
    conn.commit(); conn.close(); return jsonify({"status": "success"})

@app.route('/api/save_record', methods=['POST'])
def save_record():
    data = request.get_json(); conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    if data.get('recordType') == 'recheck':
        cursor.execute("UPDATE records SET loop_status = 'closed' WHERE id = ?", (data.get('parentRecordId'),))
    cursor.execute('''INSERT INTO records (username, time, field_name, field_internal_id, image_base64, pest_count, risk, advice, operation, record_type, parent_record_id, scheduled_recheck_time, loop_status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''', 
        (data.get('username'), data.get('time'), data.get('fieldName'), data.get('fieldId'), data.get('imageBase64'), data.get('pestCount'), data.get('risk'), data.get('advice'), data.get('operation'), data.get('recordType'), data.get('parentRecordId'), data.get('scheduledRecheckTime'), data.get('loopStatus')))
    conn.commit(); conn.close(); return jsonify({"status": "success"})

@app.route('/api/get_records', methods=['GET'])
def get_records():
    un = request.args.get('username'); conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute("SELECT id, field_name, time, image_base64, pest_count, risk, advice, operation, record_type, loop_status FROM records WHERE username = ? ORDER BY id ASC", (un,))
    rows = cursor.fetchall(); conn.close()
    return jsonify([{"id": r[0], "fieldName": r[1], "time": r[2], "imageBase64": r[3], "pestCount": r[4], "risk": r[5], "advice": r[6], "operation": r[7], "recordType": r[8], "loopStatus": r[9]} for r in rows])

# ==================== 3. 视觉核心 (极速 & 极敏版) ====================
@app.route('/api/detect', methods=['POST'])
def detect_pest():
    if 'file' not in request.files: return jsonify({"status": "error", "message": "无文件"})
    file = request.files['file']; temp_path = "temp_process.jpg"
    try:
        file.save(temp_path); img = cv2.imread(temp_path); orig = img.copy()
        h, w = img.shape[:2]; xf, yf = w / 640.0, h / 640.0
        
        # 理顺内存并做预处理
        blob = cv2.resize(img, (640, 640))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
        blob = np.ascontiguousarray(blob).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        outs = session.run([output_name], {input_name: blob})[0]
        preds = np.squeeze(outs).T if outs.shape[1] < outs.shape[2] else np.squeeze(outs)

        boxes, scores, cids = [], [], []
        for row in preds:
            s_arr = row[4:]; max_s = np.max(s_arr)
            # 🚀 极致灵敏度：2% 置信度阈值，专治小虫子
            if max_s > 0.02: 
                cid = np.argmax(s_arr); cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                boxes.append([int((cx-bw/2)*xf), int((cy-bh/2)*yf), int(bw*xf), int(bh*yf)])
                scores.append(float(max_s)); cids.append(cid)

        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.02, 0.45)
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                b = boxes[i]; cv2.rectangle(orig, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (0, 255, 0), 2)
                results.append({"name": CLASS_NAMES.get(cids[i], "害虫"), "conf": round(scores[i], 2)})

        _, buf = cv2.imencode('.jpg', orig)
        img_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"
        del img, orig, blob; gc.collect() # 强行释放大内存
        return jsonify({"status": "success", "data": {"pest_count": len(results), "risk_level": "高风险" if len(results) > 3 else "安全", "result_image": img_b64}})
    except Exception as e: return jsonify({"status": "error", "message": str(e)})
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

# ==================== 4. 大模型对话 (超长轮询版) ====================
@app.route('/api/chat', methods=['POST'])
def chat_with_agent():
    try:
        gc.collect() # 对话前清空内存
        d = request.get_json(); msg = d.get('prompt', '') or d.get('query', '')
        if not msg: return jsonify({"status": "error", "reply": "内容为空"})

        h = {"Authorization": f"Bearer {COZE_API_TOKEN}", "Content-Type": "application/json"}
        p = {"bot_id": BOT_ID, "user_id": "pro_demo_2026", "stream": False, "additional_messages": [{"role": "user", "content": str(msg), "content_type": "text"}]}
        
        r = requests.post(CREATE_CHAT_URL, headers=h, json=p, timeout=40)
        res = r.json()
        if res.get('code') != 0: return jsonify({"status": "error", "reply": "AI大脑繁忙中..."})

        cid, cvid = res['data']['id'], res['data']['conversation_id']
        # 🚀 延长轮询：最多等 20 次，每次 2.5 秒，总计等待 50 秒
        for _ in range(20):
            time.sleep(2.5)
            sr = requests.get(f"{RETRIEVE_URL}?chat_id={cid}&conversation_id={cvid}", headers=h, timeout=15).json()
            if sr['data']['status'] == 'completed':
                mr = requests.get(f"{MESSAGE_LIST_URL}?chat_id={cid}&conversation_id={cvid}", headers=h, timeout=15).json()
                for m in mr.get('data', []):
                    if m['type'] == 'answer': return jsonify({"status": "success", "reply": m['content']})
                break
            elif sr['data']['status'] == 'failed': return jsonify({"status": "error", "reply": "诊断接口出现波动"})
            
        return jsonify({"status": "error", "reply": "AI 思考太慢啦，请点击按钮重试。"})
    except Exception as e: return jsonify({"status": "error", "reply": f"通信故障: {str(e)}"})

# 其它地块状态检查存根
@app.route('/api/check_pending', methods=['GET'])
def check_pending(): return jsonify({"has_pending": False})

if __name__ == '__main__':
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

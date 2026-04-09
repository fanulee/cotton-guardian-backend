import os
import cv2
import base64
import sqlite3
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# 🚀 引入阿里云官方 SDK
import dashscope 
from dashscope import Application

app = Flask(__name__)
# 允许跨域：这是前后端云端通信的“通行证”
CORS(app)

# 设置数据库路径（适配云端环境）
DB_PATH = os.path.join(os.getcwd(), 'cotton_platform.db')

# ========== 阿里云百炼配置 ==========
# 比赛建议：将此 Key 放在环境变量中
dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY', 'sk-248f40ab61e141beaa7b2948a68cb844')
AGENT_ID = 'db0e930d13294b4f983a7506cc81e433'

# --- 数据库初始化 ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 用户表
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL, role TEXT DEFAULT 'farmer')''')
    # 地块表
    cursor.execute('''CREATE TABLE IF NOT EXISTS fields (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, field_internal_id TEXT NOT NULL, name TEXT, risk TEXT, risk_class TEXT, latlngs TEXT, sensor_images TEXT, area REAL DEFAULT 0, crop_variety TEXT DEFAULT '', plant_date TEXT DEFAULT '')''')
    # 记录表
    cursor.execute('''CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, time TEXT, field_name TEXT, field_internal_id TEXT, image_base64 TEXT, pest_count INTEGER, risk TEXT, advice TEXT, operation TEXT, record_type TEXT DEFAULT 'initial', parent_record_id INTEGER DEFAULT 0, scheduled_recheck_time TEXT, loop_status TEXT DEFAULT 'closed')''')
    conn.commit()
    conn.close()
    print("📦 数据库初始化完成！")

# --- API 路由：用户逻辑 ---
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username, password = data.get('username'), data.get('password')
    if not username or not password: return jsonify({"status": "error", "message": "不能为空"}), 400
    role = 'admin' if username.lower() == 'admin' else ('expert' if '专家' in username else 'farmer')
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, password, role))
        conn.commit()
        return jsonify({"status": "success"})
    except: return jsonify({"status": "error", "message": "账号已存在"}), 409
    finally: conn.close()

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role FROM users WHERE username = ? AND password = ?", (data.get('username'), data.get('password')))
    user = cursor.fetchone()
    conn.close()
    if user: return jsonify({"status": "success", "username": data.get('username'), "role": user[0]})
    return jsonify({"status": "error"}), 401

# --- API 路由：地块管理 ---
@app.route('/api/get_fields', methods=['GET'])
def get_fields():
    username = request.args.get('username')
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT field_internal_id, name, risk, risk_class, latlngs, sensor_images, area, crop_variety, plant_date FROM fields WHERE username = ?", (username,))
    rows = cursor.fetchall()
    conn.close()
    fields = [{"id": r[0], "name": r[1], "risk": r[2], "riskClass": r[3], "latlngs": json.loads(r[4]), "sensorImages": json.loads(r[5]), "area": r[6], "cropVariety": r[7], "plantDate": r[8]} for r in rows]
    return jsonify(fields)

@app.route('/api/save_field', methods=['POST'])
def save_field():
    data = request.get_json()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM fields WHERE username = ? AND field_internal_id = ?", (data['username'], data['id']))
    exists = cursor.fetchone()
    l_json, s_json = json.dumps(data.get('latlngs', [])), json.dumps(data.get('sensorImages', []))
    if exists:
        cursor.execute("UPDATE fields SET name=?, risk=?, risk_class=?, latlngs=?, sensor_images=?, area=?, crop_variety=?, plant_date=? WHERE username=? AND field_internal_id=?",
                       (data['name'], data['risk'], data['riskClass'], l_json, s_json, data.get('area',0), data.get('cropVariety',''), data.get('plantDate',''), data['username'], data['id']))
    else:
        cursor.execute("INSERT INTO fields (username, field_internal_id, name, risk, risk_class, latlngs, sensor_images, area, crop_variety, plant_date) VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (data['username'], data['id'], data['name'], data['risk'], data['riskClass'], l_json, s_json, data.get('area',0), data.get('cropVariety',''), data.get('plantDate','')))
    conn.commit()
    conn.close()
    return jsonify({"status": "success"})

# --- API 路由：AI 诊断 (YOLO) ---
print("正在加载模型...")
model = YOLO('best.pt') 

@app.route('/api/detect', methods=['POST'])
def detect_pest():
    if 'file' not in request.files: return jsonify({"status": "error", "message": "无文件"})
    file = request.files['file']
    temp_path = "temp_upload.jpg"
    try:
        file.save(temp_path)
        results = model(temp_path, conf=0.25, imgsz=640) # 针对云端微调参数
        res_img = results[0].plot()
        _, buffer = cv2.imencode('.jpg', res_img)
        img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        detected_items = [{"name": model.names[int(box.cls[0])], "confidence": round(float(box.conf[0]), 3)} for box in results[0].boxes]
        return jsonify({
            "status": "success",
            "data": {
                "pest_count": len(detected_items),
                "details": detected_items,
                "risk_level": "高风险" if len(detected_items) > 5 else "安全",
                "result_image": f"data:image/jpeg;base64,{img_base64}"
            }
        })
    except Exception as e: return jsonify({"status": "error", "message": str(e)})
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

# --- API 路由：智能体对话 (百炼) ---
@app.route('/api/chat', methods=['POST'])
def chat_with_agent():
    try:
        data = request.get_json()
        response = Application.call(app_id=AGENT_ID, prompt=data.get('prompt', ''))
        if response.status_code == 200:
            return jsonify({"status": "success", "reply": response.output.text})
        return jsonify({"status": "error", "reply": "智能体响应失败"})
    except Exception as e: return jsonify({"status": "error", "reply": str(e)})

# --- 启动服务 ---
if __name__ == '__main__':
    init_db()
    # 适配 Render：读取环境变量中的端口，并监听所有 IP (0.0.0.0)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
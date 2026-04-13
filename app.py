import os
import cv2
import base64
import sqlite3
import json
import time
import numpy as np
import requests
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS

# 1. 初始化 Flask 服务
app = Flask(__name__)
CORS(app)

# ================= 扣子 (Coze) 智能体核心配置 =================
COZE_API_TOKEN = 'pat_4LoBL7qWrtuswQ8zuwAMkFXwkVy4ht7pFyDtkEpMQyt2fOeDkCFRoJu3JIDp8meD'
BOT_ID = '7628183682519908395'
CREATE_CHAT_URL = "https://api.coze.cn/v3/chat"
RETRIEVE_URL = "https://api.coze.cn/v3/chat/retrieve"
MESSAGE_LIST_URL = "https://api.coze.cn/v3/chat/message/list"

# ================= 硬件优化：ONNX 加载 =================
print("正在加载轻量化 YOLO ONNX 模型...")
try:
    session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    CLASS_NAMES = {0: "棉铃虫", 1: "蚜虫", 2: "红蜘蛛"} 
    print("✅ ONNX 引擎启动成功！")
except Exception as e:
    print(f"❌ 模型启动失败: {e}")

# 设置数据库路径
DB_PATH = os.path.join(os.getcwd(), 'cotton_platform.db')

# --- 数据库初始化 ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL, role TEXT DEFAULT 'farmer')''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS fields (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, field_internal_id TEXT NOT NULL, name TEXT, risk TEXT, risk_class TEXT, latlngs TEXT, sensor_images TEXT, area REAL DEFAULT 0, crop_variety TEXT DEFAULT '', plant_date TEXT DEFAULT '')''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, time TEXT, field_name TEXT, field_internal_id TEXT, image_base64 TEXT, pest_count INTEGER, risk TEXT, advice TEXT, operation TEXT, record_type TEXT DEFAULT 'initial', parent_record_id INTEGER DEFAULT 0, scheduled_recheck_time TEXT, loop_status TEXT DEFAULT 'closed')''')
    conn.commit()
    conn.close()

# ==================== 1. 账号与鉴权路由 ====================

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    user, pwd = data.get('username'), data.get('password')
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, 'farmer')", (user, pwd))
        conn.commit()
        return jsonify({"status": "success"})
    except sqlite3.IntegrityError:
        return jsonify({"status": "error", "message": "该用户名已被注册，请换一个"})
    finally:
        conn.close()

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role FROM users WHERE username = ? AND password = ?", (data.get('username'), data.get('password')))
    user = cursor.fetchone()
    conn.close()
    if user: return jsonify({"status": "success", "username": data.get('username'), "role": user[0]})
    return jsonify({"status": "error", "message": "账号或密码错误"}), 401


# ==================== 2. 地块管理路由 ====================

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
    latlngs_json, sensors_json = json.dumps(data.get('latlngs', [])), json.dumps(data.get('sensorImages', []))
    area, crop_variety, plant_date = data.get('area', 0), data.get('cropVariety', ''), data.get('plantDate', '')
    if exists:
        cursor.execute("UPDATE fields SET name=?, risk=?, risk_class=?, latlngs=?, sensor_images=?, area=?, crop_variety=?, plant_date=? WHERE username=? AND field_internal_id=?", (data['name'], data['risk'], data['riskClass'], latlngs_json, sensors_json, area, crop_variety, plant_date, data['username'], data['id']))
    else:
        cursor.execute("INSERT INTO fields (username, field_internal_id, name, risk, risk_class, latlngs, sensor_images, area, crop_variety, plant_date) VALUES (?,?,?,?,?,?,?,?,?,?)", (data['username'], data['id'], data['name'], data['risk'], data['riskClass'], latlngs_json, sensors_json, area, crop_variety, plant_date))
    conn.commit()
    conn.close()
    return jsonify({"status": "success"})

@app.route('/api/delete_field', methods=['DELETE'])
def delete_field():
    username = request.args.get('username')
    field_id = request.args.get('id')
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM fields WHERE username = ? AND field_internal_id = ?", (username, field_id))
    conn.commit()
    conn.close()
    return jsonify({"status": "success"})


# ==================== 3. 档案闭环与复检路由 ====================

@app.route('/api/check_pending', methods=['GET'])
def check_pending():
    username = request.args.get('username')
    field_id = request.args.get('field_id')
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 查找该地块最新的一条尚未闭环的任务
    cursor.execute("SELECT id, pest_count, operation FROM records WHERE username = ? AND field_internal_id = ? AND loop_status = 'pending' ORDER BY id DESC LIMIT 1", (username, field_id))
    row = cursor.fetchone()
    conn.close()
    if row:
        return jsonify({"has_pending": True, "pending_record": {"id": row[0], "pestCount": row[1], "operation": row[2]}})
    return jsonify({"has_pending": False})

@app.route('/api/save_record', methods=['POST'])
def save_record():
    data = request.get_json()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 如果是复检记录，需要把旧任务的 loop_status 标记为 closed
    if data.get('recordType') == 'recheck':
        cursor.execute("UPDATE records SET loop_status = 'closed' WHERE id = ?", (data.get('parentRecordId'),))

    cursor.execute('''INSERT INTO records 
        (username, time, field_name, field_internal_id, image_base64, pest_count, risk, advice, operation, record_type, parent_record_id, scheduled_recheck_time, loop_status) 
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''', 
        (data.get('username'), data.get('time'), data.get('fieldName'), data.get('fieldId'), data.get('imageBase64'), data.get('pestCount'), data.get('risk'), data.get('advice'), data.get('operation'), data.get('recordType'), data.get('parentRecordId'), data.get('scheduledRecheckTime'), data.get('loopStatus'))
    )
    conn.commit()
    conn.close()
    return jsonify({"status": "success"})

@app.route('/api/get_records', methods=['GET'])
def get_records():
    username = request.args.get('username')
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM records WHERE username = ? ORDER BY id ASC", (username,))
    columns = [column[0] for column in cursor.description]
    rows = cursor.fetchall()
    conn.close()
    
    # 将下划线命名转为驼峰命名供前端使用
    records = [dict(zip(columns, row)) for row in rows]
    formatted_records = []
    for r in records:
        formatted_records.append({
            "id": r['id'], "fieldName": r['field_name'], "time": r['time'], "imageBase64": r['image_base64'],
            "pestCount": r['pest_count'], "risk": r['risk'], "advice": r['advice'], "operation": r['operation'],
            "recordType": r['record_type'], "loopStatus": r['loop_status']
        })
    return jsonify(formatted_records)


# ==================== 4. AI 视觉与对话核心引擎 ====================

@app.route('/api/detect', methods=['POST'])
def detect_pest():
    if 'file' not in request.files: return jsonify({"status": "error", "message": "无文件"})
    file = request.files['file']
    temp_path = "temp_upload.jpg"
    try:
        file.save(temp_path)
        img = cv2.imread(temp_path)
        original_img = img.copy()
        img_resized = cv2.resize(img, (640, 640))
        img_in = img_resized.transpose((2, 0, 1))[::-1]
        img_in = np.expand_dims(img_in, axis=0).astype(np.float32) / 255.0

        outputs = session.run([output_name], {input_name: img_in})[0]
        
        predictions = np.squeeze(outputs).T
        scores = np.max(predictions[:, 4:], axis=1)
        mask = scores > 0.25
        valid_preds = predictions[mask]
        
        detected_items = []
        for p in valid_preds:
            cls_id = int(np.argmax(p[4:]))
            conf = float(np.max(p[4:]))
            detected_items.append({"name": CLASS_NAMES.get(cls_id, "害虫"), "confidence": round(conf, 3)})

        _, buffer = cv2.imencode('.jpg', original_img)
        img_data_url = f"data:image/jpeg;base64,{base64.b64encode(buffer.tobytes()).decode('utf-8')}"
        
        return jsonify({
            "status": "success", 
            "data": {
                "pest_count": len(detected_items), 
                "details": detected_items, 
                "risk_level": "高风险" if len(detected_items) > 5 else "安全", 
                "result_image": img_data_url
            }
        })
    except Exception as e: return jsonify({"status": "error", "message": str(e)})
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@app.route('/api/chat', methods=['POST'])
def chat_with_agent():
    try:
        data = request.get_json()
        user_message = data.get('prompt', '') or data.get('query', '')
        
        if not user_message:
            return jsonify({"status": "error", "reply": "请输入您的问题。"})

        headers = {
            "Authorization": f"Bearer {COZE_API_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "bot_id": BOT_ID,
            "user_id": "web_user_2026",
            "stream": False,
            "additional_messages": [
                {"role": "user", "content": str(user_message), "content_type": "text"}
            ]
        }
        
        print("👉 正在呼叫扣子智能体...")
        response = requests.post(CREATE_CHAT_URL, headers=headers, json=payload, timeout=30)
        res_json = response.json()
        
        if res_json.get('code') != 0:
            return jsonify({"status": "error", "reply": f"接入错误: {res_json.get('msg')}"})

        chat_id = res_json['data']['id']
        conv_id = res_json['data']['conversation_id']

        max_retries = 30
        while max_retries > 0:
            status_url = f"{RETRIEVE_URL}?chat_id={chat_id}&conversation_id={conv_id}"
            status_resp = requests.get(status_url, headers=headers).json()
            
            if status_resp.get('code') != 0:
                return jsonify({"status": "error", "reply": f"状态查询失败: {status_resp.get('msg')}"})
            
            status = status_resp['data']['status']
            
            if status == 'completed':
                msg_list_url = f"{MESSAGE_LIST_URL}?chat_id={chat_id}&conversation_id={conv_id}"
                msg_resp = requests.get(msg_list_url, headers=headers).json()
                
                for msg in msg_resp.get('data', []):
                    if msg['type'] == 'answer':
                        return jsonify({"status": "success", "reply": msg['content']})
                break
            elif status == 'failed':
                return jsonify({"status": "error", "reply": "诊断过程发生错误，请重试。"})
            
            time.sleep(2)
            max_retries -= 1

        return jsonify({"status": "error", "reply": "AI 诊断超时，请稍后刷新重试。"})

    except Exception as e:
        print(f"❌ 后端代码崩溃: {str(e)}")
        return jsonify({"status": "error", "reply": f"系统连接故障: {str(e)}"})


if __name__ == '__main__':
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

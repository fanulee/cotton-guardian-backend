import os
import cv2
import base64
import sqlite3
import json
import time
import requests  # 用于调用扣子 API
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# 1. 初始化 Flask 服务
app = Flask(__name__)
CORS(app)

# ================= 扣子 (Coze) 智能体核心配置 =================
# 已更新为你提供的有效凭证
COZE_API_TOKEN = 'pat_4LoBL7qWrtuswQ8zuwAMkFXwkVy4ht7pFyDtkEpMQyt2fOeDkCFRoJu3JIDp8meD'
BOT_ID = '7628183682519908395'

# 扣子 V3 接口地址
CREATE_CHAT_URL = "https://api.coze.cn/v3/chat"
RETRIEVE_URL = "https://api.coze.cn/v3/chat/retrieve"
MESSAGE_LIST_URL = "https://api.coze.cn/v3/chat/message/list"
# ============================================================

# --- 数据库初始化 ---
def init_db():
    conn = sqlite3.connect('cotton_platform.db')
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

init_db()

# --- 加载 YOLO 模型 ---
print("正在加载 YOLO 模型...")
model = YOLO('best.pt') 
print("模型加载完成！")

# ==================== 业务逻辑路由 ====================

@app.route('/api/get_fields', methods=['GET'])
def get_fields():
    username = request.args.get('username')
    conn = sqlite3.connect('cotton_platform.db')
    cursor = conn.cursor()
    cursor.execute("SELECT field_internal_id, name, risk, risk_class, latlngs, sensor_images, area, crop_variety, plant_date FROM fields WHERE username = ?", (username,))
    rows = cursor.fetchall()
    conn.close()
    fields = [{"id": r[0], "name": r[1], "risk": r[2], "riskClass": r[3], "latlngs": json.loads(r[4]), "sensorImages": json.loads(r[5]), "area": r[6], "cropVariety": r[7], "plantDate": r[8]} for r in rows]
    return jsonify(fields)

@app.route('/api/save_field', methods=['POST'])
def save_field():
    data = request.get_json()
    conn = sqlite3.connect('cotton_platform.db')
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

@app.route('/api/detect', methods=['POST'])
def detect_pest():
    if 'file' not in request.files: return jsonify({"status": "error", "message": "无文件"})
    file = request.files['file']
    temp_path = "temp_upload.jpg"
    try:
        file.save(temp_path)
        results = model(temp_path, conf=0.01, imgsz=1280)
        res_img = results[0].plot()
        _, buffer = cv2.imencode('.jpg', res_img)
        img_data_url = f"data:image/jpeg;base64,{base64.b64encode(buffer.tobytes()).decode('utf-8')}"
        detected_items = [{"name": model.names[int(box.cls[0])], "confidence": round(float(box.conf[0]), 3)} for box in results[0].boxes]
        return jsonify({"status": "success", "data": {"pest_count": len(detected_items), "details": detected_items, "inference_time": round(results[0].speed['inference'], 2), "risk_level": "高风险" if len(detected_items) > 5 else "安全", "result_image": img_data_url}})
    except Exception as e: return jsonify({"status": "error", "message": str(e)})
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    conn = sqlite3.connect('cotton_platform.db')
    cursor = conn.cursor()
    cursor.execute("SELECT role FROM users WHERE username = ? AND password = ?", (data.get('username'), data.get('password')))
    user = cursor.fetchone()
    conn.close()
    if user: return jsonify({"status": "success", "username": data.get('username'), "role": user[0]})
    return jsonify({"status": "error", "message": "账号或密码错误"}), 401

# ============================================================
# 🚀 重点：更新后的扣子 (Coze) 智能体对话接口
# ============================================================
@app.route('/api/chat', methods=['POST'])
def chat_with_agent():
    try:
        data = request.get_json()
        # 前端传来的字段可能是 'prompt'
        user_message = data.get('prompt', '') or data.get('query', '')
        
        if not user_message:
            return jsonify({"status": "error", "reply": "请输入您的问题。"})

        headers = {
            "Authorization": f"Bearer {COZE_API_TOKEN}",
            "Content-Type": "application/json"
        }

        # 1. 发起对话请求 (非流式)
        payload = {
            "bot_id": BOT_ID,
            "user_id": "web_user_2026",
            "stream": False,
            "additional_messages": [
                {"role": "user", "content": str(user_message), "content_type": "text"}
            ]
        }
        
        print(f"👉 正在呼叫扣子智能体...")
        response = requests.post(CREATE_CHAT_URL, headers=headers, json=payload, timeout=30)
        res_json = response.json()
        
        if res_json.get('code') != 0:
            return jsonify({"status": "error", "reply": f"接入错误: {res_json.get('msg')}"})

        # ✨ 获取关键 ID 用于轮询
        chat_id = res_json['data']['id']
        conv_id = res_json['data']['conversation_id']

        # 2. 轮询对话状态 (等待 AI 检索知识库并生成报告)
        max_retries = 30 # 最多等待约 60 秒
        while max_retries > 0:
            status_url = f"{RETRIEVE_URL}?chat_id={chat_id}&conversation_id={conv_id}"
            status_resp = requests.get(status_url, headers=headers).json()
            
            if status_resp.get('code') != 0:
                return jsonify({"status": "error", "reply": f"状态查询失败: {status_resp.get('msg')}"})
            
            status = status_resp['data']['status']
            print(f"⏳ 植保大脑分析中: {status}")
            
            if status == 'completed':
                # 3. 诊断完成，获取消息列表提取回答
                msg_list_url = f"{MESSAGE_LIST_URL}?chat_id={chat_id}&conversation_id={conv_id}"
                msg_resp = requests.get(msg_list_url, headers=headers).json()
                
                # 在消息列表中寻找助手的正式回复内容 (type='answer')
                for msg in msg_resp.get('data', []):
                    if msg['type'] == 'answer':
                        print("✅ 决策报告已送达！")
                        return jsonify({
                            "status": "success",
                            "reply": msg['content']
                        })
                break
            elif status == 'failed':
                return jsonify({"status": "error", "reply": "诊断过程发生错误，请重试。"})
            
            time.sleep(2) # 每 2 秒检查一次
            max_retries -= 1

        return jsonify({"status": "error", "reply": "AI 诊断超时，请稍后刷新重试。"})

    except Exception as e:
        print(f"❌ 后端代码崩溃: {str(e)}")
        return jsonify({"status": "error", "reply": f"系统连接故障: {str(e)}"})

# ============================================================

if __name__ == '__main__':
    print("🚀 智棉云枢后端服务已启动！监听端口 5000")
    print("💬 扣子 RAG 智能体接口已集成。")
    app.run(host='127.0.0.1', port=5000, debug=True)

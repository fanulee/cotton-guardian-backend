import os, cv2, base64, sqlite3, json, time, requests, gc
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.onnx')
yolo_net = None
if os.path.exists(MODEL_PATH):
    try:
        yolo_net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        print("✅ 成功加载真实 YOLO 模型: best.onnx")
    except Exception as e:
        print(f"❌ 加载 YOLO 模型失败: {e}")

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
    session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    # 获取模型输入的宽高
    model_h = session.get_inputs()[0].shape[2]
    model_w = session.get_inputs()[0].shape[3]
    CLASS_NAMES = {0: "棉铃虫", 1: "蚜虫", 2: "红蜘蛛"} 
    print(f"✅ 引擎就绪！模型输入尺寸: {model_w}x{model_h}")
except Exception as e:
    print(f"❌ 引擎初始化失败: {e}")

DB_PATH = os.path.join(os.getcwd(), 'cotton_platform.db')

def init_db():
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL, role TEXT DEFAULT 'farmer')''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS fields (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, field_internal_id TEXT NOT NULL, name TEXT, risk TEXT, risk_class TEXT, latlngs TEXT, sensor_images TEXT, area REAL DEFAULT 0, crop_variety TEXT DEFAULT '', plant_date TEXT DEFAULT '')''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, time TEXT, field_name TEXT, field_internal_id TEXT, image_base64 TEXT, pest_count INTEGER, risk TEXT, advice TEXT, operation TEXT, record_type TEXT DEFAULT 'initial', parent_record_id INTEGER DEFAULT 0, scheduled_recheck_time TEXT, loop_status TEXT DEFAULT 'closed')''')
    conn.commit(); conn.close()

# ==================== 接口区 ====================

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json(); user, pwd = data.get('username'), data.get('password')
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (user, pwd))
        conn.commit(); return jsonify({"status": "success"})
    except: return jsonify({"status": "error", "message": "用户名已占用"})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json(); conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute("SELECT role FROM users WHERE username = ? AND password = ?", (data.get('username'), data.get('password')))
    user = cursor.fetchone(); conn.close()
    if user: return jsonify({"status": "success", "username": data.get('username'), "role": user[0]})
    return jsonify({"status": "error", "message": "账号或密码错误"}), 401

@app.route('/api/get_fields', methods=['GET'])
def get_fields():
    un = request.args.get('username'); conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute("SELECT field_internal_id, name, risk, risk_class, latlngs, sensor_images, area, crop_variety, plant_date FROM fields WHERE username = ?", (un,))
    rows = cursor.fetchall(); conn.close()
    return jsonify([{"id": r[0], "name": r[1], "risk": r[2], "riskClass": r[3], "latlngs": json.loads(r[4]), "sensorImages": json.loads(r[5]), "area": r[6], "cropVariety": r[7], "plantDate": r[8]} for r in rows])

@app.route('/api/save_field', methods=['POST'])
def save_field():
    d = request.get_json(); conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    lat_j, sen_j = json.dumps(d.get('latlngs', [])), json.dumps(d.get('sensorImages', []))
    cursor.execute("INSERT OR REPLACE INTO fields (username, field_internal_id, name, risk, risk_class, latlngs, sensor_images, area, crop_variety, plant_date) VALUES (?,?,?,?,?,?,?,?,?,?)", (d['username'], d['id'], d['name'], d['risk'], d['riskClass'], lat_j, sen_j, d.get('area', 0), d.get('cropVariety', ''), d.get('plantDate', '')))
    conn.commit(); conn.close(); return jsonify({"status": "success"})

# ==================== 视觉核心 (全自适应解析版) ====================
# ================= 替换您的 detect 接口 =================
@app.route('/api/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    try:
        # 1. 读取图片
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        pest_count = 0

        # 2. 真实 YOLO ONNX 模型推理
        if yolo_net is not None:
            # YOLOv8 预处理 (按 640x640)
            blob = cv2.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True, crop=False)
            yolo_net.setInput(blob)
            outputs = yolo_net.forward()
            
            # 提取预测结果
            outputs = np.transpose(np.squeeze(outputs))
            rows = outputs.shape[0]

            boxes = []
            scores = []
            
            h, w = img.shape[:2]
            x_factor = w / 640.0
            y_factor = h / 640.0

            for i in range(rows):
                classes_scores = outputs[i][4:]
                max_score = np.amax(classes_scores)
                if max_score >= 0.25: # 置信度阈值
                    x, y, bw, bh = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                    left = int((x - bw / 2) * x_factor)
                    top = int((y - bh / 2) * y_factor)
                    width = int(bw * x_factor)
                    height = int(bh * y_factor)
                    
                    boxes.append([left, top, width, height])
                    scores.append(float(max_score))

            # 非极大值抑制，去除重叠框
            indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)
            
            if len(indices) > 0:
                pest_count = len(indices)
                for i in indices.flatten():
                    box = boxes[i]
                    left, top, width, height = box[0], box[1], box[2], box[3]
                    # 绘制真实模型给出的红色边界框
                    cv2.rectangle(img, (left, top), (left + width, top + height), (0, 0, 255), 2)
                    cv2.putText(img, "Pest", (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            # 防御机制：如果模型没加载成功，为了不让前端死机，给一个错误提示
            return jsonify({"status": "error", "message": "YOLO 引擎未挂载！"}), 500

        # 3. 图片转回 Base64 发给前端
        import base64
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        risk_level = "高风险" if pest_count > 5 else "安全"
        
        return jsonify({
            "status": "success",
            "data": {
                "pest_count": pest_count,
                "risk_level": risk_level,
                "result_image": f"data:image/jpeg;base64,{img_base64}"
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route('/api/chat', methods=['POST'])
def chat_with_agent():
    try:
        gc.collect(); d = request.get_json(); msg = d.get('prompt', '') or d.get('query', '')
        if not msg: return jsonify({"status": "error", "reply": "内容为空"})
        h = {"Authorization": f"Bearer {COZE_API_TOKEN}", "Content-Type": "application/json"}
        p = {"bot_id": BOT_ID, "user_id": "pro_demo", "stream": False, "additional_messages": [{"role": "user", "content": str(msg), "content_type": "text"}]}
        
        r = requests.post(CREATE_CHAT_URL, headers=h, json=p, timeout=40)
        res = r.json()
        if res.get('code') != 0: return jsonify({"status": "error", "reply": "AI暂离，请重试"})
        
        cid, cvid = res['data']['id'], res['data']['conversation_id']
        for _ in range(25): # 增加等待次数
            time.sleep(2.5)
            sr = requests.get(f"{RETRIEVE_URL}?chat_id={cid}&conversation_id={cvid}", headers=h, timeout=15).json()
            if sr['data']['status'] == 'completed':
                mr = requests.get(f"{MESSAGE_LIST_URL}?chat_id={cid}&conversation_id={cvid}", headers=h, timeout=15).json()
                for m in mr.get('data', []):
                    if m['type'] == 'answer': return jsonify({"status": "success", "reply": m['content']})
                break
        return jsonify({"status": "error", "reply": "AI响应较慢，请稍后刷新重试"})
    except Exception as e: return jsonify({"status": "error", "reply": f"通信故障: {str(e)}"})

# 其它接口存根保持
@app.route('/api/check_pending', methods=['GET'])
def check_pending():
    username = request.args.get('username')
    if not username: return jsonify({"has_pending": False, "data": []})
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 🚨 查找这个农户名下，状态为 'pending' (待复检) 的任务
        cursor.execute("SELECT * FROM records WHERE username = ? AND loop_status = 'pending'", (username,))
        rows = cursor.fetchall()
        conn.close()
        
        pending_list = [dict(row) for row in rows]
        return jsonify({
            "has_pending": len(pending_list) > 0, 
            "data": pending_list
        })
    except Exception as e:
        print("检查复检任务失败:", e)
        return jsonify({"has_pending": False, "data": []})

@app.route('/api/save_record', methods=['POST'])
def save_record():
    try:
        d = request.get_json()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # 🚨 这里是真正的数据库写入操作
        cursor.execute('''INSERT INTO records (username, time, field_name, field_internal_id, image_base64, pest_count, risk, advice, operation, record_type, parent_record_id, scheduled_recheck_time, loop_status) 
                          VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''', 
                       (d.get('username'), d.get('time'), d.get('fieldName'), d.get('fieldId'), d.get('imageBase64'), d.get('pestCount'), d.get('risk'), d.get('advice'), d.get('operation'), d.get('recordType','initial'), d.get('parentRecordId',0), d.get('scheduledRecheckTime','7天后'), d.get('loopStatus','closed')))
        conn.commit()
        conn.close()
        return jsonify({"status": "success"})
    except Exception as e:
        print("归档写入失败:", e) # 在 Render 日志里打印真实错误
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/get_records', methods=['GET'])
def get_records():
    # 1. 获取前端传来的用户名
    username = request.args.get('username')
    if not username:
        return jsonify([]) # 如果没传用户名，才返回空

    try:
        # 2. 连接数据库并查询这个用户的所有记录
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # 这一行让返回的数据像字典一样可以通过列名访问
        cursor = conn.cursor()
        
        # 按照时间倒序查询（最新的在最前面）
        cursor.execute("SELECT * FROM records WHERE username = ? ORDER BY id DESC", (username,))
        rows = cursor.fetchall()
        conn.close()

        # 3. 将数据库查到的数据打包成列表发回给前端
        records_list = []
        for row in rows:
            records_list.append(dict(row))
            
        return jsonify(records_list)
        
    except Exception as e:
        print("读取档案失败:", e)
        return jsonify([]) # 如果数据库崩了，为了不让前端卡死，返回空


@app.route('/api/delete_field', methods=['DELETE'])
def delete_field(): return jsonify({"status": "success"})

if __name__ == '__main__':
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

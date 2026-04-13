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
@app.route('/api/detect', methods=['POST'])
def detect_pest():
    if 'file' not in request.files: return jsonify({"status": "error", "message": "无文件"})
    file = request.files['file']; temp_path = "temp_u.jpg"
    try:
        file.save(temp_path); img = cv2.imread(temp_path); orig = img.copy()
        h, w = img.shape[:2]
        
        # 1. 预处理
        blob = cv2.resize(img, (640, 640))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
        blob = np.ascontiguousarray(blob).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        # 2. 推理
        outs = session.run([output_name], {input_name: blob})[0]
        
        # 3. 动态矩阵解析 (核心改进)
        # 如果是 (1, 7, 8400) 格式，需要转置
        if outs.shape[1] < outs.shape[2]: 
            preds = np.squeeze(outs).T
        else: 
            preds = np.squeeze(outs)

        boxes, scores, cids = [], [], []
        # 计算缩放因子
        xf, yf = w / 640.0, h / 640.0

        for row in preds:
            # 前4位是坐标，后面是所有类别的得分
            # 有些模型输出包含 objectness，所以我们要动态取类别得分
            prob_scores = row[4:] 
            max_s = np.max(prob_scores)
            
            # 🚀 极致灵敏：即使是很小的虫子也能被捕捉
            if max_s > 0.02: 
                cid = np.argmax(prob_scores)
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                
                # 转换到原图坐标
                l = int((cx - bw/2) * xf)
                t = int((cy - bh/2) * yf)
                rw = int(bw * xf)
                rh = int(bh * yf)
                
                boxes.append([l, t, rw, rh])
                scores.append(float(max_s))
                cids.append(cid)

        # 4. 非极大值抑制 (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.02, 0.45)
        
        results = []
        if len(indices) > 0:
            # 兼容不同版本的 NMS 返回值
            idx_list = indices.flatten() if hasattr(indices, 'flatten') else indices
            for i in idx_list:
                b = boxes[i]
                # 5. 在大图上画框
                cv2.rectangle(orig, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (0, 255, 0), 3)
                # 写上文字
                label = f"{CLASS_NAMES.get(cids[i], 'Bug')}:{scores[i]:.2f}"
                cv2.putText(orig, label, (b[0], b[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                results.append({"name": CLASS_NAMES.get(cids[i], "害虫"), "conf": round(scores[i], 2)})

        _, buf = cv2.imencode('.jpg', orig)
        img_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"
        
        del img, orig, blob; gc.collect()
        return jsonify({
            "status": "success", 
            "data": {
                "pest_count": len(results), 
                "risk_level": "高风险" if len(results) > 3 else "安全", 
                "result_image": img_b64
            }
        })
    except Exception as e: return jsonify({"status": "error", "message": str(e)})
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

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
def check_pending(): return jsonify({"has_pending": False})

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

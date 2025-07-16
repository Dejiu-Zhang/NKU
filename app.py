import os
import json
import numpy as np
import faiss
from openai import OpenAI
from flask import Flask, request, render_template, jsonify, session
from datetime import datetime
import uuid

# ✅ 初始化 Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your-secret-key-here")  # 用于 session

# ✅ 初始化 OpenAI 客户端
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ✅ 检查文件是否存在
import os
print("当前目录文件列表：", os.listdir('.'))

# ✅ 加载嵌入向量与 chunk 文本
# 处理 JSONL 格式文件
id2chunks = {}
jsonl_file = "nku_chunks_multilevel.jsonl"
npz_file = "nku_memory_vectors_full.npz"

# 检查文件是否存在
if not os.path.exists(jsonl_file):
    print(f"错误：找不到文件 {jsonl_file}")
    exit(1)

if not os.path.exists(npz_file):
    print(f"错误：找不到文件 {npz_file}")
    exit(1)

# 读取 JSONL 文件
try:
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if line.strip():  # 跳过空行
                try:
                    data = json.loads(line)
                    # 使用行号作为 ID，或者使用数据中的 id 字段
                    chunk_id = str(line_num) if 'id' not in data else str(data['id'])
                    # 假设每行包含 'text' 或 'content' 字段
                    chunk_text = data.get('text', data.get('content', line.strip()))
                    id2chunks[chunk_id] = chunk_text
                except json.JSONDecodeError:
                    print(f"警告：第{line_num}行JSON解析失败，跳过")
                    continue
    print(f"成功加载 {len(id2chunks)} 个文档块")
except Exception as e:
    print(f"读取JSONL文件失败：{e}")
    exit(1)

# 读取嵌入向量 (.npz 格式)
try:
    with np.load(npz_file) as data:
        # 打印 npz 文件中的所有数组名称
        print(f"NPZ文件中的数组：{list(data.keys())}")
        
        # 通常 npz 文件中的数组可能叫 'arr_0', 'embeddings', 'vectors' 等
        # 我们需要找到正确的数组名称
        if 'arr_0' in data:
            all_embeddings = data['arr_0']
        elif 'embeddings' in data:
            all_embeddings = data['embeddings']
        elif 'vectors' in data:
            all_embeddings = data['vectors']
        else:
            # 如果不确定名称，就使用第一个数组
            array_name = list(data.keys())[0]
            all_embeddings = data[array_name]
            print(f"使用数组：{array_name}")
            
    print(f"成功加载嵌入向量，形状：{all_embeddings.shape}")
except Exception as e:
    print(f"读取NPZ文件失败：{e}")
    exit(1)

chunk_ids = list(id2chunks.keys())
embedding_dim = all_embeddings.shape[1]

# ✅ 构建 FAISS 索引
index = faiss.IndexFlatL2(embedding_dim)
index.add(all_embeddings)

# ✅ 在内存中存储会话历史（生产环境建议使用数据库）
chat_sessions = {}

def get_session_id():
    """获取或创建会话ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def get_chat_history(session_id):
    """获取聊天历史"""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    return chat_sessions[session_id]

def add_to_chat_history(session_id, user_message, assistant_response, retrieved_chunks):
    """添加到聊天历史"""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    chat_sessions[session_id].append({
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'user_message': user_message,
        'assistant_response': assistant_response,
        'retrieved_chunks': retrieved_chunks
    })

def retrieve_relevant_chunks(user_query, k=5):
    """检索相关文档块"""
    # 🔹 向量化用户问题
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[user_query]
    )
    query_embed = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
    
    # 🔹 FAISS 检索
    distances, indices = index.search(query_embed, k)
    
    retrieved_chunks = []
    for idx in indices[0]:
        chunk_id = chunk_ids[idx]
        retrieved_chunks.append(id2chunks[chunk_id])
    
    return retrieved_chunks

def generate_response(user_query, chat_history, retrieved_chunks):
    """生成回答"""
    # 🔹 构造 context
    context = "\n\n".join(retrieved_chunks)
    
    # 🔹 准备消息历史
    messages = [
        {
            "role": "system", 
            "content": "你是一个熟悉南开大学新生指南的校园助手，请根据提供的内容用中文简洁明了地回答问题。请保持对话的连贯性，可以参考之前的对话内容。"
        }
    ]
    
    # 🔹 添加历史对话（最近5轮，避免 token 过多）
    recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
    for chat in recent_history:
        messages.append({"role": "user", "content": chat['user_message']})
        messages.append({"role": "assistant", "content": chat['assistant_response']})
    
    # 🔹 构造当前问题的 prompt
    current_prompt = (
        f"以下是南开大学新生指南中的相关内容：\n{context}\n\n"
        f"请根据上述信息回答这个问题：{user_query}\n"
        f"⚠️ 注意：尽量直接使用原文中的表述，不要用你自己的语言改写。如果能原封不动引用就尽量引用，回答越详细越好。"
    )
    
    messages.append({"role": "user", "content": current_prompt})
    
    # 🔹 生成回答
    chat_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3
    )
    
    return chat_response.choices[0].message.content.strip()

# ✅ 首页：聊天界面
@app.route("/")
def index():
    return render_template("chat.html")

# ✅ 获取聊天历史
@app.route("/api/history")
def get_history():
    session_id = get_session_id()
    history = get_chat_history(session_id)
    return jsonify(history)

# ✅ 发送消息
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_query = data.get("message", "").strip()
        
        if not user_query:
            return jsonify({"error": "消息不能为空"}), 400
        
        session_id = get_session_id()
        chat_history = get_chat_history(session_id)
        
        # 🔹 检索相关内容
        retrieved_chunks = retrieve_relevant_chunks(user_query)
        
        # 🔹 生成回答
        assistant_response = generate_response(user_query, chat_history, retrieved_chunks)
        
        # 🔹 添加到历史记录
        add_to_chat_history(session_id, user_query, assistant_response, retrieved_chunks)
        
        return jsonify({
            "response": assistant_response,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "处理请求时出现错误"}), 500

# ✅ 清空聊天历史
@app.route("/api/clear", methods=["POST"])
def clear_history():
    session_id = get_session_id()
    if session_id in chat_sessions:
        chat_sessions[session_id] = []
    return jsonify({"message": "聊天历史已清空"})

# ✅ 启动
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port)

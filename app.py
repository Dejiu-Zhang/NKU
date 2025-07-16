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

# ✅ 加载嵌入向量与 chunk 文本
id2chunks = {}
chunk_ids = []
jsonl_file = "nku_chunks_multilevel.jsonl"
npz_file = "nku_memory_vectors_full.npz"

# 检查文件是否存在
if not os.path.exists(npz_file):
    print(f"错误：找不到文件 {npz_file}")
    exit(1)

# 首先尝试从 NPZ 文件中读取所有数据
try:
    with np.load(npz_file) as data:
        print(f"NPZ文件中的数组：{list(data.keys())}")
        
        # 读取嵌入向量
        if 'all_embeddings' in data:
            all_embeddings = data['all_embeddings']
            print("使用数组：all_embeddings")
        elif 'embeddings' in data:
            all_embeddings = data['embeddings']
            print("使用数组：embeddings")
        elif 'vectors' in data:
            all_embeddings = data['vectors']
            print("使用数组：vectors")
        else:
            # 查找二维数组（嵌入向量应该是二维的）
            found_embedding = False
            for array_name in data.keys():
                if len(data[array_name].shape) == 2:
                    all_embeddings = data[array_name]
                    print(f"使用二维数组：{array_name}")
                    found_embedding = True
                    break
            
            if not found_embedding:
                print("错误：找不到合适的嵌入向量数组")
                exit(1)
        
        # 读取chunk_ids（用于索引）
        if 'chunk_ids' in data:
            chunk_ids = data['chunk_ids'].tolist()
            print(f"从NPZ文件读取到 {len(chunk_ids)} 个chunk_ids")
        
        # 尝试从 NPZ 文件中读取 id2chunks
        if 'id2chunks' in data:
            try:
                stored_id2chunks = data['id2chunks'].item()  # .item() 用于读取字典
                print(f"从NPZ文件读取到 {len(stored_id2chunks)} 个文档块")
                id2chunks = {str(k): str(v) for k, v in stored_id2chunks.items()}
            except Exception as e:
                print(f"NPZ中的id2chunks读取失败: {e}，将从JSONL文件读取")
                
    print(f"成功加载嵌入向量，形状：{all_embeddings.shape}")
except Exception as e:
    print(f"读取NPZ文件失败：{e}")
    exit(1)

# 如果没有chunk_ids，从id2chunks的键生成
if len(chunk_ids) == 0:
    chunk_ids = list(id2chunks.keys())
    print(f"从id2chunks生成chunk_ids: {len(chunk_ids)}")

# 如果从 NPZ 文件中没有成功读取到 id2chunks，则从 JSONL 文件读取
if len(id2chunks) == 0:
    print("从JSONL文件读取文档块...")
    if not os.path.exists(jsonl_file):
        print(f"错误：找不到文件 {jsonl_file}")
        exit(1)
    
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
        print(f"从JSONL文件成功加载 {len(id2chunks)} 个文档块")
        
        # 如果chunk_ids为空，从新读取的id2chunks生成
        if len(chunk_ids) == 0:
            chunk_ids = list(id2chunks.keys())
            print(f"从JSONL生成chunk_ids: {len(chunk_ids)}")
    except Exception as e:
        print(f"读取JSONL文件失败：{e}")
        exit(1)

# 验证数据一致性
print(f"最终数据:")
print(f"- 文档块数量: {len(id2chunks)}")
print(f"- chunk_ids数量: {len(chunk_ids)}")
print(f"- 嵌入向量形状: {all_embeddings.shape}")

# 检查索引一致性
if len(chunk_ids) != all_embeddings.shape[0]:
    print(f"警告：chunk_ids数量({len(chunk_ids)})与嵌入向量数量({all_embeddings.shape[0]})不匹配")
    # 尝试修复
    if len(chunk_ids) > all_embeddings.shape[0]:
        chunk_ids = chunk_ids[:all_embeddings.shape[0]]
        print(f"截断chunk_ids到{len(chunk_ids)}个")
    else:
        print("错误：无法修复索引不匹配问题")
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
    try:
        print(f"开始向量化查询: {user_query}")
        # 🔹 向量化用户问题
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[user_query]
        )
        print("向量化完成")
        
        query_embed = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
        print(f"查询向量形状: {query_embed.shape}")
        
        # 🔹 FAISS 检索
        print("开始FAISS检索...")
        distances, indices = index.search(query_embed, k)
        print(f"检索完成，找到 {len(indices[0])} 个结果")
        
        retrieved_chunks = []
        for idx in indices[0]:
            chunk_id = chunk_ids[idx]
            if chunk_id in id2chunks:
                retrieved_chunks.append(id2chunks[chunk_id])
            else:
                print(f"警告: chunk_id {chunk_id} 不在 id2chunks 中")
        
        print(f"成功检索到 {len(retrieved_chunks)} 个文档块")
        return retrieved_chunks
        
    except Exception as e:
        print(f"检索过程出错: {e}")
        import traceback
        print(f"检索错误详情: {traceback.format_exc()}")
        return []

def generate_response(user_query, chat_history, retrieved_chunks):
    """生成回答"""
    try:
        print("开始生成回答...")
        # 🔹 构造 context
        context = "\n\n".join(retrieved_chunks)
        print(f"上下文长度: {len(context)}")
        
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
            f"⚠️ 注意：尽量直接使用原文中的表述，不要用你自己的语言改写，回答越详细越好。但是！你绝对不要瞎编！！"
        )
        
        messages.append({"role": "user", "content": current_prompt})
        print(f"消息数量: {len(messages)}")
        
        # 🔹 生成回答
        print("调用OpenAI API...")
        chat_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )
        print("OpenAI API调用成功")
        
        response_text = chat_response.choices[0].message.content.strip()
        print(f"生成回答长度: {len(response_text)}")
        return response_text
        
    except Exception as e:
        print(f"生成回答出错: {e}")
        import traceback
        print(f"生成回答错误详情: {traceback.format_exc()}")
        return "抱歉，生成回答时出现错误，请稍后再试。"

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
        print("=== 开始处理聊天请求 ===")
        data = request.get_json()
        print(f"接收到的数据: {data}")
        
        user_query = data.get("message", "").strip()
        print(f"用户查询: {user_query}")
        
        if not user_query:
            return jsonify({"error": "消息不能为空"}), 400
        
        session_id = get_session_id()
        print(f"会话ID: {session_id}")
        
        chat_history = get_chat_history(session_id)
        print(f"历史记录数量: {len(chat_history)}")
        
        # 🔹 检索相关内容
        print("开始检索相关内容...")
        retrieved_chunks = retrieve_relevant_chunks(user_query)
        print(f"检索到 {len(retrieved_chunks)} 个相关块")
        
        # 🔹 生成回答
        print("开始生成回答...")
        assistant_response = generate_response(user_query, chat_history, retrieved_chunks)
        print(f"生成的回答长度: {len(assistant_response)}")
        
        # 🔹 添加到历史记录
        print("添加到历史记录...")
        add_to_chat_history(session_id, user_query, assistant_response, retrieved_chunks)
        
        print("=== 聊天请求处理完成 ===")
        return jsonify({
            "response": assistant_response,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"聊天处理错误: {e}")
        import traceback
        print(f"完整错误信息: {traceback.format_exc()}")
        return jsonify({"error": f"处理请求时出现错误: {str(e)}"}), 500

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

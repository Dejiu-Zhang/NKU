import os
import json
import numpy as np
import faiss
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

# ========= 配置区 =========
EMBEDDING_PATH = "nku_memory_vectors_full.npz"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"
TOP_K = 6
# ==========================

client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ 加载向量库
data = np.load(EMBEDDING_PATH, allow_pickle=True)
chunk_ids = data["chunk_ids"]
all_embeddings = data["all_embeddings"]
id2chunks = data["id2chunks"].item()  # 注意：是纯字符串

# ✅ 构建 FAISS 索引
d = all_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(all_embeddings)

# ✅ Flask app
app = Flask(__name__)

# ✅ 路由：主页
@app.route("/")
def index():
    return render_template("index.html")  # 需要有 templates/index.html 页面

# ✅ 路由：Chat API
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        user_query = request.json.get("query", "").strip()
        if not user_query:
            return jsonify({"error": "Empty query"}), 400

        # Step 1: 向量化
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[user_query]
        )
        query_vec = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)

        # Step 2: 检索相似文本
        D, I = index.search(query_vec, TOP_K)
        retrieved_chunks = [id2chunks[chunk_ids[i]] for i in I[0]]
        context = "\n---\n".join(retrieved_chunks)

        # Step 3: 构造 prompt
        prompt = f"""你是一个热心的南开大学新生向导助手。以下是与用户问题最相关的手册内容：

{context}

现在，请根据上述信息回答用户的问题：

{user_query}
请尽量直接使用原文中的表述，尽量不要用你自己的语言改写。"""

        # Step 4: 调用 GPT 模型回答
        reply = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = reply.choices[0].message.content.strip()
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": f"❌ 出错啦：{str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

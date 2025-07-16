import os
import json
import numpy as np
import faiss
from flask import Flask, request, render_template
from openai import OpenAI

# ✅ 初始化 Flask 应用
app = Flask(__name__)

# ✅ 初始化 OpenAI 客户端（读取环境变量中的 API Key）
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ✅ 加载向量嵌入和 chunk 映射
embedding_path = "nku_memory_vectors_full.npz"
data = np.load(embedding_path, allow_pickle=True)
chunk_ids = data["chunk_ids"]
all_embeddings = data["all_embeddings"]
id2chunks = data["id2chunks"].item()

# ✅ 构建 FAISS 向量索引
embedding_dim = all_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(all_embeddings)

# ✅ 检索函数：根据用户 query 返回最相似的记忆块
def retrieve_chunks(query, top_k=6):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    query_vec = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
    D, I = faiss_index.search(query_vec, top_k)
    return [id2chunks[chunk_ids[i]] for i in I[0]]

# ✅ 问答函数：调用 GPT-4o-mini 给出回答
def answer_question(query):
    chunks = retrieve_chunks(query, top_k=6)
    context = "\n---\n".join(chunks)
    prompt = f"""你是一个热心的南开大学新生向导助手。以下是与用户问题最相关的手册内容：

{context}

请根据以上内容，回答用户的问题：{query}。尽量使用原文中的语言进行引用。"""

    reply = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return reply.choices[0].message.content.strip(), chunks

# ✅ 主路由（避免与 faiss_index 重名，改为 home）
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form["user_input"]
        try:
            answer, refs = answer_question(user_input)
        except Exception as e:
            answer = f"❌ 出错啦：{str(e)}"
            refs = []
        return render_template("index.html", user_input=user_input, answer=answer, refs=refs)
    return render_template("index.html", user_input="", answer="", refs=[])

# ✅ 启动服务（支持本地调试和 Render 部署）
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

import os
import json
import numpy as np
import faiss
from flask import Flask, request, render_template
from openai import OpenAI

# ✅ 初始化 Flask 应用
app = Flask(__name__)

# ✅ 读取 OpenAI API 密钥（Render 中配置 OPENAI_API_KEY 环境变量）
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ✅ 加载 embedding 文件
embedding_path = "nku_memory_vectors_full.npz"
data = np.load(embedding_path, allow_pickle=True)
chunk_ids = data["chunk_ids"]
all_embeddings = data["all_embeddings"]
id2chunks = data["id2chunks"].item()

# ✅ FAISS 索引构建
d = all_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(all_embeddings)

# ✅ 检索函数
def retrieve_chunks(query, top_k=6):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    query_vec = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
    D, I = index.search(query_vec, top_k)
    return [id2chunks[chunk_ids[i]] for i in I[0]]

# ✅ Chatbot 回答函数
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

# ✅ 首页路由
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]
        try:
            answer, refs = answer_question(user_input)
        except Exception as e:
            answer = f"❌ 出错啦：{str(e)}"
            refs = []
        return render_template("chat.html", user_input=user_input, answer=answer, refs=refs)
    return render_template("chat.html", user_input="", answer="", refs=[])

# ✅ 启动服务
if __name__ == "__main__":
    app.run(debug=True)

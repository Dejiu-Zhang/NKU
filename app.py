import os
import json
import numpy as np
import faiss
from openai import OpenAI
from flask import Flask, request, render_template

# ✅ 初始化 Flask app
app = Flask(__name__)

# ✅ 初始化 OpenAI 客户端（建议设置环境变量 OPENAI_API_KEY）
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ✅ 加载嵌入向量与 chunk 文本
with open("id2chunks.json", "r", encoding="utf-8") as f:
    id2chunks = json.load(f)

with open("id2embeds.npy", "rb") as f:
    all_embeddings = np.load(f)

chunk_ids = list(id2chunks.keys())
embedding_dim = all_embeddings.shape[1]

# ✅ 构建 FAISS 索引
index = faiss.IndexFlatL2(embedding_dim)
index.add(all_embeddings)

# ✅ 首页：提问界面
@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    retrieved_chunks = []
    
    if request.method == "POST":
        user_query = request.form.get("query")

        # 🔹 向量化用户问题
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[user_query]
        )
        query_embed = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)

        # 🔹 FAISS 检索
        k = 5
        distances, indices = index.search(query_embed, k)
        for idx in indices[0]:
            chunk_id = chunk_ids[idx]
            retrieved_chunks.append(id2chunks[chunk_id])

        # 🔹 构造 Prompt
        context = "\n\n".join(retrieved_chunks)
        system_prompt = "你是一个熟悉南开大学新生指南的校园助手，请根据提供的内容用中文简洁明了地回答问题。"
        final_prompt = (
            f"以下是南开大学新生指南中的相关内容：\n{context}\n\n"
            f"请根据上述信息回答这个问题：{user_query}。\n"
            f"⚠️ 注意：尽量直接使用原文中的表述，不要用你自己的语言改写。如果能原封不动引用就尽量引用，回答越详细越好。"
        )

        # 🔹 生成回答
        chat_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.3
        )

        answer = chat_response.choices[0].message.content.strip()

    return render_template("index.html", answer=answer)

# ✅ 启动
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7860)

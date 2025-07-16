import os
import json
import numpy as np
import faiss
from flask import Flask, request, render_template
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

embedding_path = "nku_memory_vectors_full.npz"
data = np.load(embedding_path, allow_pickle=True)
chunk_ids = data["chunk_ids"]
all_embeddings = data["all_embeddings"]
id2chunks = data["id2chunks"].item()

embedding_dim = all_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(all_embeddings)

def retrieve_chunks(query, top_k=6):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    query_vec = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
    D, I = faiss_index.search(query_vec, top_k)
    return [id2chunks[chunk_ids[i]] for i in I[0]]

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

from zhipuai import ZhipuAI
import chromadb
from config import ZHIPUAI_API_KEY
import fitz
import jieba
from rank_bm25 import BM25Okapi

def extract_text_from_file(uploaded_file):
    """升级版：使用 PyMuPDF 提取文本，尽可能保留原文档的段落和排版顺序"""
    text = ""
    if uploaded_file.name.endswith(".txt"):
        text = uploaded_file.getvalue().decode("utf-8")

    elif uploaded_file.name.endswith(".pdf"):
        # 将 Streamlit 上传的内存文件转交给 PyMuPDF 处理
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            # 使用 blocks 模式提取，它会将文本按自然的段落块划分
            blocks = page.get_text("blocks")
            # 简单按照 Y 轴（上下）和 X 轴（左右）对文本块进行排序，尽量解决双栏问题
            blocks.sort(key=lambda b: (b[1], b[0]))
            for b in blocks:
                # b[4] 是文本内容
                block_text = b[4].strip()
                if block_text:
                    # 替换掉多余的内部换行符，让段落更紧凑
                    block_text = block_text.replace('\n', ' ')
                    text += block_text + "\n\n"
    return text


import re


def get_text_chunks(text, chunk_size=400, chunk_overlap=50):
    """升级版：基于段落和句子的语义切片，防止把一句话切断"""
    # 1. 先按段落（双换行）将长文本拆开
    paragraphs = text.split('\n\n')

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 如果当前块加上新段落还没超标，就拼进去
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += para + "\n"
        else:
            # 如果超标了，先把当前块存起来
            if current_chunk:
                chunks.append(current_chunk.strip())

            # 如果这单独的一个段落就超过了最大长度，需要按句号强行切分
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[。！？.!?])', para)
                sub_chunk = ""
                for sentence in sentences:
                    if len(sub_chunk) + len(sentence) <= chunk_size:
                        sub_chunk += sentence
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk.strip())
                        sub_chunk = sentence
                current_chunk = sub_chunk  # 剩下的零碎句子做下一个块的开头
            else:
                # 开启新的一块，并尝试加入一定的重叠上下文（取上一个块的结尾）
                overlap_text = chunks[-1][-chunk_overlap:] if chunks and chunk_overlap > 0 else ""
                current_chunk = overlap_text + "\n" + para + "\n"

    # 把最后一个块也加进去
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# 1. 初始化智谱 AI 客户端
client = ZhipuAI(api_key=ZHIPUAI_API_KEY)

# 2. 初始化本地向量数据库
chroma_client = chromadb.PersistentClient(path="./second_brain_db")
collection = chroma_client.get_or_create_collection(name="chat_history")


def add_memory(msg_id, text, sender, timestamp):
    """将一条记录存入向量数据库"""
    response = client.embeddings.create(
        model="embedding-2",
        input=text
    )
    vector = response.data[0].embedding

    collection.add(
        embeddings=[vector],
        documents=[text],
        metadatas=[{"sender": sender, "time": timestamp}],
        ids=[msg_id]
    )


def query_memory(question, top_k=10):
    """
    混合检索版：结合 ChromaDB 的向量检索与内存中的 BM25 关键词检索，
    最后通过 RRF (倒数排名融合) 算法重新排序。
    """
    # ==========================================
    # 1. 向量检索 (Vector Search)
    # ==========================================
    response = client.embeddings.create(model="embedding-2", input=question)
    query_vector = response.data[0].embedding

    # 稍微扩大召回量，为后续的融合提供足够的候选池
    vector_results = collection.query(
        query_embeddings=[query_vector],
        n_results=20
    )

    # 提取向量检索得分靠前的 ID 列表
    vector_ranked_ids = vector_results['ids'][0] if vector_results['ids'] else []

    # ==========================================
    # 2. 关键词检索 (BM25 Lexical Search)
    # ==========================================
    # 获取数据库中的全量数据用于构建 BM25 索引
    all_data = collection.get()
    all_ids = all_data['ids']
    all_docs = all_data['documents']
    all_metas = all_data['metadatas']

    bm25_ranked_ids = []
    if all_docs:
        # 使用 jieba 对所有文档进行中文分词
        tokenized_corpus = [list(jieba.cut(doc)) for doc in all_docs]
        bm25 = BM25Okapi(tokenized_corpus)

        # 对用户问题进行分词并打分
        tokenized_query = list(jieba.cut(question))
        bm25_scores = bm25.get_scores(tokenized_query)

        # 将得分与 ID 打包，并按分数从高到低排序
        bm25_ranked = sorted(zip(all_ids, bm25_scores), key=lambda x: x[1], reverse=True)
        # 只保留分数大于 0 的有效匹配，最多取前 20 名
        bm25_ranked_ids = [item[0] for item in bm25_ranked if item[1] > 0][:20]

    # ==========================================
    # 3. RRF 倒数排名融合 (Reciprocal Rank Fusion)
    # ==========================================
    rrf_scores = {}
    k = 60  # RRF 平滑常数，经验值通常设为 60

    # 计入向量检索的 RRF 得分
    for rank, doc_id in enumerate(vector_ranked_ids):
        # 公式：1 / (k + 排名)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    # 计入 BM25 检索的 RRF 得分
    for rank, doc_id in enumerate(bm25_ranked_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    # 根据总 RRF 分数对所有候选 ID 进行最终排序，取前 top_k 个
    final_ranked_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    # ==========================================
    # 4. 组装最终结果
    # ==========================================
    formatted_memories = []
    for doc_id in final_ranked_ids:
        # 找到该 ID 在全量数据中的原始索引
        idx = all_ids.index(doc_id)
        doc = all_docs[idx]
        meta = all_metas[idx]
        formatted_memories.append(f"[{meta['time']}] {meta['sender']}: {doc}")

    return formatted_memories


def delete_memory(msg_id):
    """根据 ID 删除记忆"""
    collection.delete(ids=[msg_id])


def update_memory(msg_id, text, sender, timestamp):
    """根据 ID 更新记忆（如果 ID 存在则覆盖）"""
    # 重新生成向量
    response = client.embeddings.create(model="embedding-2", input=text)
    vector = response.data[0].embedding

    collection.update(
        ids=[msg_id],
        embeddings=[vector],
        documents=[text],
        metadatas=[{"sender": sender, "time": timestamp}]
    )
def get_all_sources():
    """获取所有唯一的记忆来源（文件列表或用户手动输入）"""
    results = collection.get(include=["metadatas"])
    sources = set()
    if results and results['metadatas']:
        for meta in results['metadatas']:
            if meta and 'sender' in meta:
                sources.add(meta['sender'])
    return sorted(list(sources))

def delete_memory_by_source(source_name):
    """根据来源名称一次性删除所有相关片段"""
    collection.delete(where={"sender": source_name})

def search_memories_in_db(query_text):
    """在数据库中进行简单的文本关键词搜索（用于管理界面）"""
    # 注意：这里使用的是 where_document 的内容包含搜索，不是向量检索
    results = collection.get(
        where_document={"$contains": query_text},
        include=["documents", "metadatas"]
    )
    return results


def delete_memory_by_keyword(keyword):
    """根据关键词进行语义搜索，并删除最相关的记忆"""
    try:
        # 1. 把 AI 提取的关键词变成向量
        response = client.embeddings.create(model="embedding-2", input=keyword)
        query_vector = response.data[0].embedding

        # 2. 去数据库查最相关的 3 条记忆（数量可以自己调）
        results = collection.query(query_embeddings=[query_vector], n_results=1)

        deleted_count = 0
        if results and results['ids'] and len(results['ids'][0]) > 0:
            ids_to_delete = results['ids'][0]
            # 3. 执行删除操作
            collection.delete(ids=ids_to_delete)
            deleted_count = len(ids_to_delete)

        return deleted_count
    except Exception as e:
        print(f"删除记忆失败: {e}")
        return 0
# 测试脚本
if __name__ == "__main__":
    print("数据库当前条数:", collection.count())
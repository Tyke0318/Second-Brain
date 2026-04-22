import streamlit as st
import uuid
from datetime import datetime
from brain import query_memory, client, add_memory, collection, extract_text_from_file, get_text_chunks, \
    search_memories_in_db, get_all_sources, delete_memory_by_source

st.set_page_config(page_title="Second Brain", layout="wide")
st.title("Second Brain")

from brain import update_memory  # 记得导入更新函数

@st.dialog("Global Memory Base Management", width="large")
def memory_management_dialog():
    """这是一个独立的弹窗视图，专门用于管理记忆"""
    # 1. 搜索框：实时过滤记忆内容
    search_query = st.text_input("Search Memory...", placeholder="Enter keywords...")

    if search_query:
        # 执行搜索模式
        search_results = search_memories_in_db(search_query)
        if search_results['ids']:
            st.caption(f"Found {len(search_results['ids'])} related record(s):")
            for i in range(len(search_results['ids'])):
                m_id = search_results['ids'][i]
                m_doc = search_results['documents'][i]
                m_meta = search_results['metadatas'][i]

                with st.expander(f"Source: {m_meta.get('sender', 'Unknown')} | Time: {m_meta.get('time', 'Unknown')}"):
                    # 🌟 核心修改 1：将只读的 st.write 变成可编辑的 st.text_area
                    edited_text = st.text_area(
                        "Edit Memory",
                        value=m_doc,
                        key=f"edit_area_{m_id}",
                        height=100,
                        label_visibility="collapsed"  # 隐藏标签名，让界面更紧凑
                    )

                    # 🌟 核心修改 2：并排提供“保存”与“删除”按钮
                    col_save, col_del, _ = st.columns([1.5, 1.5, 7])

                    with col_save:
                        if st.button("Save Changes", key=f"save_{m_id}"):
                            # 调用 brain.py 中的更新函数，保持原有的 sender 和 time 属性
                            update_memory(m_id, edited_text, m_meta.get('sender'), m_meta.get('time'))
                            st.toast("✅ Memory content updated!")
                            st.rerun()

                    with col_del:
                        if st.button("Delete", key=f"del_{m_id}"):
                            collection.delete(ids=[m_id])
                            st.toast("Memory snippet deleted!")
                            st.rerun()
        else:
            st.write("No matching content found.")

    else:
        # 默认：按来源分组管理模式
        sources = get_all_sources()
        if sources:
            for source in sources:
                with st.expander(f"{source}"):
                    col1, col2 = st.columns([4, 1])
                    col1.write(f"**Source:** {source}")
                    if col2.button("🗑️ Clear File Memory", key=f"del_src_{source}"):
                        delete_memory_by_source(source)
                        st.toast(f"Cleared source: {source}")
                        st.rerun()

                    # 展示预览 (为了性能，默认视图依然只做只读预览)
                    source_items = collection.get(where={"sender": source})
                    if source_items['documents']:
                        st.divider()
                        for idx, doc in enumerate(source_items['documents'][:5]):
                            st.caption(f"Snippet {idx + 1}: {doc[:100]}...")
                        if len(source_items['documents']) > 5:
                            st.caption(f" (... total {len(source_items['documents'])} records. To modify, please use the search bar above) ")
        else:
            st.info("Brain is currently empty~")

# === 核心修改点：在侧边栏渲染前定义回调函数 ===
def submit_memory():
    # 1. 直接通过 key ("memo_widget") 获取输入框的内容
    memo = st.session_state.memo_widget

    if memo and memo.strip():  # 确保不是纯空格或空字符串
        msg_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        add_memory(msg_id, memo, "User", timestamp)

        # 2. 彻底清理：直接将绑定的 key 重置为空字符串
        st.session_state.memo_widget = ""

        # 3. 设置一个 flag 用于触发提示
        st.session_state.show_toast = True

# --- 侧边栏 ---
with st.sidebar:
    st.header("System Control")
    st.success("AI Assistant Connected")
    try:
        st.info(f"Current memory count: {collection.count()}")
    except:
        pass

    st.divider()
    st.header("📝 Enter New Memory")

    # 使用 Tabs 将两种录入方式合并在一处
    tab_text, tab_file = st.tabs(["✍️ Short Text Entry", "📄 Long Document Upload"])

    # 定义一个统一的固定高度，比如 350px (你可以根据实际观感微调)
    CONTAINER_HEIGHT = 300

    # === 标签页 1：文本录入 ===
    with tab_text:
        # 🌟 核心修改：使用固定高度、无边框的容器包裹内容
        with st.container(height=CONTAINER_HEIGHT, border=False):
            with st.form(key="memo_form", clear_on_submit=True):
                # 顺便给 text_area 增加 height 属性，让它看起来更饱满
                memo = st.text_area("What do you need me to remember?", placeholder="e.g., My birthday is May 20th", height=180)
                submit_btn = st.form_submit_button("Save to Brain")

                if submit_btn:
                    if memo and memo.strip():
                        msg_id = str(uuid.uuid4())
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                        add_memory(msg_id, memo, "User", timestamp)
                        st.toast("Text memory permanently saved!")
                    else:
                        st.warning("Please enter valid content before submitting.")

    # === 标签页 2：文档上传 ===
    with tab_file:
        # 🌟 核心修改：使用同等高度的容器
        with st.container(height=CONTAINER_HEIGHT, border=False):
            uploaded_file = st.file_uploader("Supports TXT and PDF files", type=["txt", "pdf"])

            if st.button("Parse and Save to Brain", key="upload_doc_btn"):
                if uploaded_file is not None:
                    with st.status("Processing document...", expanded=True) as status:
                        try:
                            st.write("1/3 Extracting text...")
                            raw_text = extract_text_from_file(uploaded_file)

                            if not raw_text.strip():
                                status.update(label="Document extraction failed or is empty", state="error")
                                st.error("Failed to extract valid text from the file.")
                            else:
                                st.write("2/3 Chunking text...")
                                chunks = get_text_chunks(raw_text, chunk_size=400, chunk_overlap=50)
                                total_chunks = len(chunks)

                                st.write("3/3 Vectorizing and saving to brain...")
                                progress_bar = st.progress(0)
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                                source_name = f"Document: {uploaded_file.name}"

                                for i, chunk in enumerate(chunks):
                                    msg_id = f"{uuid.uuid4()}_chunk_{i}"

                                    # 🌟 核心优化：把来源信息硬编码进被向量化的文本中！
                                    # 这样大模型在搜 "DepMamba" 时，哪怕切片里只有 "D-Vlog"，
                                    # 也会因为这句前缀而获得极高的向量匹配分。
                                    enhanced_chunk = f"[Document Theme/Source: {uploaded_file.name}]\n{chunk}"
                                    add_memory(msg_id, enhanced_chunk, source_name, timestamp)
                                    progress_bar.progress((i + 1) / total_chunks)

                                status.update(label="Document memory saved successfully!", state="complete")
                                st.toast(f"Successfully converted {uploaded_file.name} into {total_chunks} memories!")
                        except Exception as e:
                            status.update(label="Error occurred during processing", state="error")
                            st.error(f"Error message: {str(e)}")
                else:
                    st.warning("Please select a file first!")

    # ---记忆管理区域 ---
    st.divider()
    st.header("Memory Management")
    st.write("Clear out unnecessary knowledge fragments to keep the brain clean.")

    # 放置一个醒目的入口按钮
    if st.button("Open Global Memory Manager", type="primary", use_container_width=True):
        memory_management_dialog()  # 点击后调用弹窗函数


# --- 聊天界面 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入逻辑
if prompt := st.chat_input("Ask your Second Brain..."):
    # 1. 记录并显示用户问题
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 生成回答
    with st.chat_message("assistant"):
        # A. 检索
        with st.status("Retrieving relevant memories...", expanded=False) as status:
            memories = query_memory(prompt)
            context_text = "\n".join(memories) if memories else "No relevant memories found."
            status.update(label="Retrieval complete", state="complete")

        # B. 构造提示词
        system_prompt = f"""You are an AI assistant with an eternal memory.
        Please answer the user's question based on the provided [Memory Snippets] below. If there is no relevant information in the memory, please answer based on your general knowledge, and remind the user that this is not in the memory.

        [Relevant Memories]:
        {context_text}
        """
        # C. 流式生成
        response_placeholder = st.empty()
        full_response = ""

        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
                response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
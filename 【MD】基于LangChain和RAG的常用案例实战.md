# PDF 文档问答ChatBot
## 本地上传文档
+ 支持 **pdf**
+ 支持 **txt**
+ 支持 **doc/docx**

## 问答页面
![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1724400774841-0da24c4e-9a73-4bbf-bccb-bc7f05b67832.png)

## python环境
+ 新建一个`requirements.txt`文件

```plain
streamlit
python-docx
PyPDF2
faiss-cpu
langchain
langchain-core
langchain-community
langchain-openai
```

+ 然后安装相应的包

```plain
pip install -r requirements.txt -U
```

## 代码
> 创建一个 <font style="color:#080808;background-color:#ffffff;">pdf_search.py</font> 文件， 把下边的复制进去  
注意：配置好OPEN_API 接口地址和密钥的环境变量
>

```python
#示例：pdf_search.py
# 导入Streamlit库，用于创建Web应用
import streamlit as st
# 导入递归字符文本分割器，用于将文档分割成小块
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 导入FAISS向量存储，用于存储和检索文档嵌入
from langchain_community.vectorstores import FAISS
# 导入OpenAI聊天模型
from langchain_openai import ChatOpenAI
# 导入OpenAI嵌入模型，用于生成文本嵌入
from langchain_openai import OpenAIEmbeddings
# 导入Document类，用于封装文档内容和元数据
from langchain_core.documents import Document
# 导入对话检索链，用于处理对话和检索
from langchain.chains import ConversationalRetrievalChain
# 导入docx库，用于处理Word文档
import docx
# 导入PyPDF2库，用于处理PDF文档
from PyPDF2 import PdfReader

# 设置页面配置，包括标题、图标和布局
st.set_page_config(page_title="文档问答", page_icon=":robot:", layout="wide")

# 设置页面的CSS样式
st.markdown(
    """<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
.stDeployButton {
            visibility: hidden;
        }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.block-container {
    padding: 2rem 4rem 2rem 4rem;
}

.st-emotion-cache-16txtl3 {
    padding: 3rem 1.5rem;
}
</style>
# """,
    unsafe_allow_html=True,
)

# 定义机器人消息模板
bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn.icon-icons.com/icons2/1371/PNG/512/robot02_90810.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""

# 定义用户消息模板
user_template = """
<div class="chat-message user">
    <div class="avatar">
        <img src="https://www.shareicon.net/data/512x512/2015/09/18/103160_man_512x512.png" >
    </div>    
    <div class="message">{{MSG}}</div>
</div>
"""

# 从PDF文件中提取文本
def get_pdf_text(pdf_docs):
    # 存储提取的文档
    docs = []
    for document in pdf_docs:
        if document.type == "application/pdf":
            # 读取PDF文件
            pdf_reader = PdfReader(document)
            for idx, page in enumerate(pdf_reader.pages):
                docs.append(
                    Document(
                        # 提取页面文本
                        page_content=page.extract_text(),
                        # 添加元数据
                        metadata={"source": f"{document.name} on page {idx}"},
                    )
                )
        elif document.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # 读取Word文档
            doc = docx.Document(document)
            for idx, paragraph in enumerate(doc.paragraphs):
                docs.append(
                    Document(
                        # 提取段落文本
                        page_content=paragraph.text,
                        # 添加元数据
                        metadata={"source": f"{document.name} in paragraph {idx}"},
                    )
                )
        elif document.type == "text/plain":
            # 读取纯文本文件
            text = document.getvalue().decode("utf-8")
            docs.append(Document(page_content=text, metadata={"source": document.name}))

    return docs

# 将文档分割成小块文本
def get_text_chunks(docs):
    # 创建文本分割器
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    # 分割文档
    docs_chunks = text_splitter.split_documents(docs)
    return docs_chunks

# 创建向量存储
def get_vectorstore(docs_chunks):
    # 创建OpenAI嵌入模型
    embeddings = OpenAIEmbeddings()
    # 创建FAISS向量存储
    vectorstore = FAISS.from_documents(docs_chunks, embedding=embeddings)
    return vectorstore

# 创建对话检索链
def get_conversation_chain(vectorstore):
    # 创建OpenAI聊天模型
    llm = ChatOpenAI(model="gpt-4o")
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        # 使用向量存储作为检索器
        retriever=vectorstore.as_retriever(),
        # 返回源文档
        return_source_documents=True,
    )
    return conversation_chain

# 处理用户输入并生成响应
def handle_userinput_pdf(user_question):
    # 获取聊天历史
    chat_history = st.session_state.chat_history
    # 生成响应
    response = st.session_state.conversation(
        {"question": user_question, "chat_history": chat_history}
    )
    # 添加用户问题到聊天历史
    st.session_state.chat_history.append(("user", user_question))
    # 添加机器人回答到聊天历史
    st.session_state.chat_history.append(("assistant", response["answer"]))

    # 显示用户问题
    st.write(
        user_template.replace("{{MSG}}", user_question),
        unsafe_allow_html=True,
    )

    # 获取源文档
    sources = response["source_documents"]
    # 提取源文档名称
    source_names = set([i.metadata["source"] for i in sources])
    # 合并源文档名称
    src = "\n\n".join(source_names)
    src = f"\n\n> source : {src}"
    message = st.session_state.chat_history[-1]
    # 显示机器人回答和源文档
    st.write(bot_template.replace("{{MSG}}", message[1] + src), unsafe_allow_html=True)

# 显示聊天历史记录
def show_history():
    # 获取聊天历史
    chat_history = st.session_state.chat_history
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            # 显示用户消息
            st.write(
                user_template.replace("{{MSG}}", message[1]),
                unsafe_allow_html=True,
            )
        else:
            # 显示机器人消息
            st.write(
                bot_template.replace("{{MSG}}", message[1]), unsafe_allow_html=True
            )

# 主函数
def main():
    # 显示页面标题
    st.header("Chat with Documents")
    # 初始化会话状态
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    with st.sidebar:
        # 显示侧边栏标题
        st.title("文档管理")
        # 文件上传控件
        pdf_docs = st.file_uploader(
            "选择文件",
            # 支持的文件类型
            type=["pdf", "txt", "doc", "docx"],
            # 支持多文件上传
            accept_multiple_files=True,
        )
        if st.button(
                "处理文档",
                # 设置最后操作为pdf
                on_click=lambda: setattr(st.session_state, "last_action", "pdf"),
                use_container_width=True,
        ):
            if pdf_docs:
                # 显示处理中的旋转器
                with st.spinner("Processing"):
                    # 提取PDF、doc、txt文本
                    # chatgpt.pdf 拆分为3个doc
                    # knowledge.txt 拆分为1个doc
                    # news.docx 拆分为37个doc
                    docs = get_pdf_text(pdf_docs)
                    # 分割文本
                    docs_chunks = get_text_chunks(docs)
                    # 创建向量存储
                    vectorstore = get_vectorstore(docs_chunks)
                    # 创建对话链
                    st.session_state.conversation = get_conversation_chain(vectorstore)
            else:
                # 提示用户上传文件
                st.warning("记得上传文件哦~~")

        def clear_history():
            # 清空聊天历史
            st.session_state.chat_history = []

        if st.session_state.chat_history:
            # 清空对话按钮
            st.button("清空对话", on_click=clear_history, use_container_width=True)

    with st.container():
        # 获取用户输入
        user_question = st.chat_input("输入点什么~")

    with st.container(height=400):
        # 显示聊天历史
        show_history()
        if user_question:
            if st.session_state.conversation is not None:
                # 处理用户输入
                handle_userinput_pdf(user_question)
            else:
                # 提示用户上传文件
                st.warning("记得上传文件哦~~")

# 运行主函数
if __name__ == "__main__":
    main()
```

## 启动
```powershell
streamlit run pdf_search.py
```



## 问答效果
**问题1：chatgpt为什么那么火爆?**![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1724400878852-52f054bd-2e41-4694-bc43-dbcf0ebd54d9.png)

**问题2：2024世界人工智能大会哪天开始?**

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1724401022867-ea991b96-598a-4b68-bdf8-51cd3a5d4ee5.png)

**问题3：Pixar公司是做什么的?**

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1724401096332-cce91ede-b48a-46d6-a9d9-baa2e7ea8e8b.png)

**问题4：黄河为什么被称为母亲河?**

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1724401179826-3d6ce67f-9be4-444f-af9e-5efe9fe1b67a.png)

# 基于 Web URL 的问答ChatBot
## <font style="color:rgb(51, 51, 51);">导入库</font>
<font style="color:rgb(51, 51, 51);">我们将首先导入聊天机器人所需的库。</font>

```python
#示例：web_search.py
import time
from datetime import datetime

import requests
import streamlit as st
import wikipedia
from bs4 import BeautifulSoup
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from streamlit_chat import message
global docsearch
from langchain.globals import set_verbose
docsearch = None
```

## <font style="color:rgb(51, 51, 51);">爬取维基百科</font>
<font style="color:rgb(51, 51, 51);">构建聊天机器人的第一步是访问维基百科文章并提取内容。该</font>`<font style="color:rgb(51, 51, 51);background-color:rgb(243, 244, 244);">get_wiki</font>`<font style="color:rgb(51, 51, 51);">函数接受搜索词并返回整页内容和维基百科文章的摘要。该</font>`<font style="color:rgb(51, 51, 51);background-color:rgb(243, 244, 244);">wikipedia.summary</font>`<font style="color:rgb(51, 51, 51);">方法搜索摘要，以及</font>`<font style="color:rgb(51, 51, 51);background-color:rgb(243, 244, 244);">requests</font>`<font style="color:rgb(51, 51, 51);">用于访问文章的 URL 的模块。该</font>`<font style="color:rgb(51, 51, 51);background-color:rgb(243, 244, 244);">BeautifulSoup</font>`<font style="color:rgb(51, 51, 51);">模块使用在解析页面的HTML内容，该</font>`<font style="color:rgb(51, 51, 51);background-color:rgb(243, 244, 244);">content_div.find_all('p')</font>`<font style="color:rgb(51, 51, 51);">行从页面上的段落中提取文本。</font>

```python
def get_wiki(search):
    # 将语言设置为简体中文（默认为自动检测）
    lang = "zh"

    """
    从维基百科获取摘要
    """
    # set language to zh_CN (default is auto-detect)
    wikipedia.set_lang(lang)
    summary = wikipedia.summary(search, sentences=5)

    """
    抓取所请求查询的维基百科页面
    """

    # 根据用户输入和语言创建URL
    url = f"https://{lang}.wikipedia.org/wiki/{search}"

    # 向URL发送GET请求并解析HTML内容
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # 提取页面的主要内容
    content_div = soup.find(id="mw-content-text")

    # 摘录所有内容段落
    paras = content_div.find_all('p')

    # 将段落连接成整页内容
    full_page_content = ""
    for para in paras:
        full_page_content += para.text

    # 打印整页内容
    return full_page_content, summary
```

## <font style="color:rgb(51, 51, 51);">设置用户界面</font> 
<font style="color:rgb(51, 51, 51);">接下来，我们使用 Streamlit 设置用户界面。我们首先创建一个标题：</font>

```python
st.markdown("<h1 style='text-align: center; color: Black;'>基于 Web URL 的问答</h1>", unsafe_allow_html=True)
```

<font style="color:rgb(51, 51, 51);">这将为聊天机器人创建一个大而居中的标题。</font>

<font style="color:rgb(51, 51, 51);">环境变量配置好</font><font style="color:#080808;background-color:#ffffff;">OPENAI_BASE_URL和OPENAI_API_KEY</font>

```python
setx OPENAI_BASE_URL "https://api.openai.com/v1"
setx OPENAI_API_KEY "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

<font style="color:rgb(51, 51, 51);">一旦用户输入他们的 OpenAI 密钥，我们就会初始化 GPT 模型并要求他们输入搜索查询，该查询将用于抓取相关的 Wikipedia 页面。get_wiki()函数将返回搜索查询和抓取页面的摘要。现在，如果它返回了一些值，则 Q&A 字段将被激活，用户可以提问。</font>

```python
search = st.text_input("请输入要检索的关键词")
if len(search):
    wiki_content, summary = get_wiki(search)

    if len(wiki_content):
        try:
            # 创建用户发送消息的输入文本框
            st.write(summary)
            user_query = st.text_input("You: ", "", key="input")
            send_button = st.button("Send")
```

<font style="color:rgb(51, 51, 51);">现在，我们初始化FAISS向量数据库</font>

```python
def init_db(wiki_content):
    print("初始化FAISS向量数据库...")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(wiki_content)
    embeddings = OpenAIEmbeddings()
    global doc_search
    doc_search = FAISS.from_texts(texts, embeddings)
```

<font style="color:rgb(51, 51, 51);">建立索引后，我们就可以查询用户的请求。</font>

```python
# 创建一个函数来获取机器人响应
def get_bot_response(user_query):
    # 在向量数据库中进行相似性搜索，返回6个结果
    docs = doc_search.similarity_search(user_query, K=6)
    main_content = user_query + "\n\n"
    # 拼接用户查询和相似的文本内容
    for doc in docs:
        main_content += doc.page_content + "\n\n"
    messages.append(HumanMessage(content=main_content))
    # 调用OpenAI接口获取响应
    ai_response = chat.invoke(messages).content
    # 将刚刚添加的 HumanMessage 从 messages 列表中移除。这样做的原因是，main_content 包含了用户的原始查询和相似文本内容，
    # 但在实际的对话历史中，我们只希望保留用户的原始查询和 AI 的响应，而不是包含相似文本内容的查询。
    messages.pop()
    # 将用户查询添加到消息列表
    messages.append(HumanMessage(content=user_query))
    # 将用户查询添加到消息列表
    messages.append(AIMessage(content=ai_response))
    return ai_response
```

<font style="color:rgb(51, 51, 51);">就这样！你就拥有了专属于您的友好机器人，它可以回答您关于维基百科文章的查询。</font>



## 问答效果
**问题1：黄河**

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1724398479647-ac19e53e-3bf8-48c7-aa72-1cd85392afcf.png)

**问题2：黄河为什么是世界上含沙量最高的河流?**

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1724398492510-e3cecc20-9733-4a55-8171-c2794f4b53cc.png)

# 基于 SQL 的 CSV 数据分析问答 Web Search 集成
## 需求
基于 LangChain 和 Streamlit 的 Web 应用，用于使用 LLM 和嵌入从 SQLite 数据库中搜索相关的offer。用户可以输入与品牌、类别或零售商相关的搜索查询，也支持通过SQL语句进行搜索，应用程序将从数据库中检索并显示相关的offer。该应用使用 OpenAI API 进行自然语言处理和嵌入生成。

SQLite 官网：[https://www.sqlite.org/pragma.html#toc](https://www.sqlite.org/pragma.html#toc)

SQLite 使用手册：[https://www.runoob.com/sqlite/sqlite-select.html](https://www.runoob.com/sqlite/sqlite-select.html)



## 方法
+ **目标**：该方法的目标是基于产品类别、品牌或零售商查询从 `offer_retailer` 表中提取相关的offer。鉴于所需数据分散在 `data` 目录中的多个表中，采用了语言模型（LLM）来促进智能数据库查询。
+ **数据库准备**：最初，使用存储在 `data` 目录中的 `.csv` 文件构建了一个本地 SQLite 数据库。这是通过 `sqlite3` 和 `pandas` 库实现的。
+ **LLM 集成**：通过 `langchain_experimental.sql.SQLDatabaseChain` 实现了语言模型（LLM）与本地 SQLite 数据库的有效交互。
+ **提示工程**：该方法的一个重要方面是制定合适的提示，以指导 LLM 最佳地检索和格式化数据库条目。通过多次迭代和实验来微调这个提示。
+ **相似度评分**：为了确定检索结果与查询的相关性，进行了余弦相似度比较。使用 `langchain_openai.OpenAIEmbeddings` 生成嵌入进行比较，从而对结果进行排序。
+ **Streamlit 集成**：最后一步是解析 LLM 的输出，并围绕它构建一个用户友好的 Streamlit 应用，允许用户进行交互式搜索。

## 环境
在开始之前，请确保满足以下要求：

+ Python 3.12.4 或更高版本
+ OpenAI API 密钥
+ 包含以下表的 SQLite 数据库：`brand_category`，`categories` 和 `offer_retailer`

安装所需的包：

```bash
pip install -r requirements.txt
```

确保您的 SQLite 数据库已设置好，并包含必要的表（`brand_category`，`categories`，`offer_retailer`）。



<font style="color:#DF2A3F;">注意：streamlit版本需要</font><font style="color:#DF2A3F;background-color:#ffffff;"><1.30，一般为1.29.0，否则启动会报以下错误</font>

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1729647910330-b2757326-ba90-4ecf-adcd-c8c5e6cf721f.png)



## 代码
```python
#示例：csv_search.py
import os
# 导入正则表达式模块
import re
import sqlite3
import pandas as pd
import streamlit as st
from llm import RetrievalLLM

# 数据文件路径
DATA_PATH = 'data'
# 数据表名称
TABLES = ('brand_category', 'categories', 'offer_retailer')
# 数据库名称
DB_NAME = 'offer_db.sqlite'
# 提示模板
PROMPT_TEMPLATE = """
                你会接收到一个查询，你的任务是从`offer_retailer`表中的`OFFER`字段检索相关offer。
                查询可能是混合大小写的，所以也要搜索大写版本的查询。
                重要的是，你可能需要使用数据库中其他表的信息，即：`brand_category`, `categories`, `offer_retailer`，来检索正确的offer。
                不要虚构offer。如果在`offer_retailer`表中找不到offer，返回字符串：`NONE`。
                如果你能从`offer_retailer`表中检索到offer，用分隔符`#`分隔每个offer。例如，输出应该是这样的：`offer1#offer2#offer3`。
                如果SQLResult为空，返回`None`。不要生成任何offer。
                这是查询：`{}`
                """

# Streamlit应用标题
st.title("搜索offer 🔍")

# 连接SQLite数据库
conn = sqlite3.connect('offer_db.sqlite')

# 判断是否是SQL查询的函数
def is_sql_query(query):
    # 定义一个包含常见 SQL 关键字的列表
    sql_keywords = [
        'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER',
        'TRUNCATE', 'MERGE', 'CALL', 'EXPLAIN', 'DESCRIBE', 'SHOW'
    ]

    # 去掉查询字符串两端的空白字符并转换为大写
    query_upper = query.strip().upper()

    # 遍历 SQL 关键字列表
    for keyword in sql_keywords:
        # 如果查询字符串以某个 SQL 关键字开头，返回 True
        if query_upper.startswith(keyword):
            return True

    # 定义一个正则表达式模式，用于匹配以 SQL 关键字开头的字符串
    sql_pattern = re.compile(
        r'^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|TRUNCATE|MERGE|CALL|EXPLAIN|DESCRIBE|SHOW)\s+',
        re.IGNORECASE  # 忽略大小写
    )

    # 如果正则表达式匹配查询字符串，返回 True
    if sql_pattern.match(query):
        return True

    # 如果查询字符串不符合任何 SQL 关键字模式，返回 False
    return False


# 创建一个表单用于搜索
with st.form("search_form"):
    # 输入框用于输入查询
    query = st.text_input("通过类别、品牌或发布商搜索offer。")
    # 提交按钮
    submitted = st.form_submit_button("搜索")
    # 实例化RetrievalLLM类
    retrieval_llm = RetrievalLLM(
        data_path=DATA_PATH,
        tables=TABLES,
        db_name=DB_NAME,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
    )
    # 如果表单提交
    if submitted:
        # 如果输入内容是SQL语句，则显示SQL执行结果
        if is_sql_query(query):
            st.write(pd.read_sql_query(query, conn))
        # 否则使用LLM从数据库中检索offer
        else:
            # 使用RetrievalLLM实例检索offer
            retrieved_offers = retrieval_llm.retrieve_offers(
                PROMPT_TEMPLATE.format(query)
            )
            # 如果没有找到相关offer
            if not retrieved_offers:
                st.text("未找到相关offer。")
            else:
                # 显示检索到的offer
                st.table(retrieval_llm.parse_output(retrieved_offers, query))

```



 

```python
#示例：llm.py
import sqlite3
import numpy as np
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain


class RetrievalLLM:
    """一个类，用于使用大型语言模型（LLM）检索和重新排序offer。

    参数:
        data_path (str): 包含数据CSV文件的目录路径。
        tables (list[str]): 数据CSV文件的名称列表。
        db_name (str): 用于存储数据的SQLite数据库名称。
        openai_api_key (str): OpenAI API密钥。

    属性:
        data_path (str): 包含数据CSV文件的目录路径。
        tables (list[str]): 数据CSV文件的名称列表。
        db_name (str): 用于存储数据的SQLite数据库名称。
        openai_api_key (str): OpenAI API密钥。
        db (SQLDatabase): SQLite数据库连接。
        llm (OpenAI): OpenAI LLM客户端。
        embeddings (OpenAIEmbeddings): OpenAI嵌入客户端。
        db_chain (SQLDatabaseChain): 与LLM集成的SQL数据库链。
    """

    def __init__(self, data_path, tables, db_name, openai_api_key):
        # 初始化类属性
        self.data_path = data_path
        self.tables = tables
        self.db_name = db_name
        self.openai_api_key = openai_api_key

        # 读取CSV文件并存储到数据帧字典中
        dfs = {}
        for table in self.tables:
            dfs[table] = pd.read_csv(f"{self.data_path}/{table}.csv")

        # 将数据帧写入SQLite数据库
        with sqlite3.connect(self.db_name) as local_db:
            for table, df in dfs.items():
                df.to_sql(table, local_db, if_exists="replace")

        # 创建SQL数据库连接
        self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_name}")
        # 创建OpenAI LLM客户端
        self.llm = OpenAI(
            temperature=0, verbose=True, openai_api_key=self.openai_api_key
        )
        # 创建OpenAI嵌入客户端
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        # 创建SQL数据库链
        self.db_chain = SQLDatabaseChain.from_llm(self.llm, self.db)
        self.allow_reuse = True

    def retrieve_offers(self, prompt):
        """使用LLM从数据库中检索offer。

        参数:
            prompt (str): 用于检索offer的提示。

        返回:
            list[str]: 检索到的offer列表。
        """

        # 运行SQL数据库链以检索offer
        retrieved_offers = self.db_chain.run(prompt)
        # 如果retrieved_offers是"None"，则返回None，否则返回检索到的offer
        return None if retrieved_offers == "None" else retrieved_offers

    def get_embeddings(self, documents):
        """使用LLM获取文档的嵌入。

        参数:
            documents (list[str]): 文档列表。

        返回:
            np.ndarray: 包含文档嵌入的NumPy数组。
        """

        # 如果文档列表只有一个文档，将单个文档的嵌入转换为Numpy数组
        if len(documents) == 1:
            return np.asarray(self.embeddings.embed_query(documents[0]))
        else:
            # 否则获取每个文档的嵌入并存储到列表中
            embeddings_list = []
            for document in documents:
                embeddings_list.append(self.embeddings.embed_query(document))
            return np.asarray(embeddings_list)

    def parse_output(self, retrieved_offers, query):
        """解析retrieve_offers()方法的输出并返回一个数据帧。

        参数:
            retrieved_offers (list[str]): 检索到的offer列表。
            query (str): 用于检索offer的查询。

        返回:
            pd.DataFrame: 包含匹配相似度和offer的数据帧。
        """

        # 分割检索到的offer
        top_offers = retrieved_offers.split("#")

        # 获取查询的嵌入
        query_embedding = self.get_embeddings([query])
        # 获取offer的嵌入
        offer_embeddings = self.get_embeddings(top_offers)

        # offer_embeddings是一个二维的Numpy数组，包含多个offer的嵌入向量。
        # query_embedding是一个二维的Numpy数组，包含查询的嵌入向量。
        # query_embedding.T是查询嵌入的转置，使其成为一个列向量，便于进行矩阵乘法。
        # np.dot()计算每个offer嵌入向量与查询嵌入向量之间的点积（内积），结果是一个二维数组，其中每个元素表示一个offer与查询之间的相似度分数。
        # .flatten() 将二维数组转换为一维数组，得到每个 offer 与查询之间的相似度分数列表。
        sim_scores = np.dot(offer_embeddings, query_embedding.T).flatten()
        # 计算相似度得分，转换为百分比形式
        sim_scores = [p * 100 for p in sim_scores]

        # 创建数据帧并按相似度排序
        df = (
            pd.DataFrame({"匹配相似度 %": sim_scores, "offer": top_offers})
            .sort_values(by=["匹配相似度 %"], ascending=False)
            .reset_index(drop=True)
        )
        df.index += 1
        return df

```

## 运行
本地运行应用

```bash
streamlit  csv_search.py
```

应用运行后，打开浏览器并导航到 `http://localhost:8501` 访问offer搜索界面。

1. 在文本输入框中输入您的搜索查询（品牌、类别或零售商）。
2. 点击“搜索”按钮启动搜索。
3. 匹配查询的相关offer将以表格形式显示。



## 问答效果
**问题1：select * from categories**

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1724401368100-c44f11a3-5be1-434c-b721-e2ada0608fe0.png)

**问题2：select CATEGORY_ID from categories**

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1724401403726-e204fe12-e3e0-4832-964b-e042e934a77d.png)

**问题3：RED GOLD**

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1724405445966-2d9be43d-89e6-44b0-bb4f-82d27529ec12.png)


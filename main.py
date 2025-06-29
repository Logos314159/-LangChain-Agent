from langchain_community.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
import requests

# --------------- 配置环境变量 --------------
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("请设置环境变量 OPENAI_API_KEY")

# --------------- 初始化知识库向量 -------------
if not os.path.exists("chroma_db"):
    print(">>> 初始化知识库向量...")
    with open("data/project_doc.md", "r", encoding="utf-8") as f:
        text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_text(text)
    doc_objs = [Document(page_content=chunk) for chunk in docs]
    vectordb = Chroma.from_documents(doc_objs, embedding=OpenAIEmbeddings(), persist_directory="chroma_db")
    vectordb.persist()
else:
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=OpenAIEmbeddings())

def knowledge_lookup(query: str) -> str:
    result = vectordb.similarity_search(query, k=2)
    return result[0].page_content if result else "未找到答案"

# --------------- 天气工具 -------------------
def get_weather(city: str) -> str:
    api_key = "605e406768866873775bd4873acb8fb4"  # 这里记得替换成你注册的
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        resp = requests.get(url)
        data = resp.json()
        if data.get("main"):
            return f"{city}当前温度: {data['main']['temp']}°C"
        else:
            return f"{city}天气查询失败"
    except Exception as e:
        return f"查询异常: {e}"

# --------------- 本地函数 --------------------
def sort_and_average(nums: str) -> str:
    try:
        numlist = [float(x) for x in nums.split()]
        sorted_nums = sorted(numlist)
        avg = sum(numlist)/len(numlist)
        return f"排序: {sorted_nums}, 平均值: {avg:.2f}"
    except Exception as e:
        return f"输入异常: {e}"

# --------------- 注册工具 --------------------
tools = [
    Tool(
        name="KnowledgeBase",
        func=knowledge_lookup,
        description="基于文档的问答"
    ),
    Tool(
        name="WeatherQuery",
        func=get_weather,
        description="输入城市名查询天气"
    ),
    Tool(
        name="SortAndAverage",
        func=sort_and_average,
        description="输入数字并以空格分隔，返回排序与均值"
    )
]

# --------------- Agent --------------------
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    memory=memory,
    verbose=True
)

if __name__ == "__main__":
    print("🤖 One-Man Agent 启动，输入 `exit` 退出。")
    while True:
        user = input("\n🧑 你: ")
        if user.lower() in ["exit", "quit"]:
            break
        reply = agent.run(user)
        print(f"🤖 Agent: {reply}")

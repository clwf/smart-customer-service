import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from knowledge_base import KnowledgeBase
import re

load_dotenv()
kb = KnowledgeBase(docs_dir="./documents")
# 创建 AI 客户端
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0,
)
class State(TypedDict):
    message: str
    emotion: str
    need_human: bool
    intention: str
    info: str
    result: str
    rounds: int
    quality: bool
    memory: list

def extract_expression(text: str) -> str:
    pattern = r'[\d\+\-\*/\(\)\.]+'
    match = re.search(pattern, text)
    if match:
        return match.group().strip()
    else:
        return ""   # 未找到表达式

def analyze_emotion(state: State) -> dict:
    message = state["message"]
    result = llm.invoke([
        SystemMessage(content=(
            "你是一个情感分析专家。"
            "只回复一个词：正面、负面或中性。"
        )),
        HumanMessage(content=message),
    ])
    sentence = result.content.strip()
    #负面转人工
    if sentence == "负面":
        need_human = True
    else:
        need_human = False
    return {
        "emotion": sentence,
        "need_human": need_human,
    }

def intent_recognition(state: State) -> dict:
    message = state["message"]
    result = llm.invoke([
        SystemMessage(content=(
            "你是一个意图识别专家。"
            "意图只分三类：company_policy（公司制度）、calculate（计算）、chat（闲聊）"
        )),
        HumanMessage(content=message),
    ])
    sentence = result.content.strip()
    return {
        "intention": sentence,
    }

def human_intervention(state: State) -> dict:
    return {"result": "非常抱歉给您带来不好的体验，已为您转接人工客服，请稍候..."}



def route_after_emotion(state: State) -> str:
    if state["need_human"]:
        return "to_human"          # 任意 key
    else:
        return "to_intent"         # 任意 key




def knowledge_retrieval(state: State) -> dict:
    message = state["message"]
    result = kb.search(message)
    return {"info": result}


def calculator(state: State) -> dict:
    message = state["message"]
    expression = extract_expression(message)
    if not expression:
        return {"info": "未能从消息中提取到数学表达式"}
    try:
        import numexpr
        result = numexpr.evaluate(expression)
        return {"info": str(result)}
    except Exception as e:
        return {"info": f"计算出错：{str(e)}"}


def direct_answer(state: State) -> dict:
    message = state["message"]
    memory = state["memory"]

    messages = [SystemMessage(content="你是一个友好的助手，请简洁回答问题。")]

    # 只取最近4条消息（2轮对话），避免历史太长
    recent = memory[-4:] if len(memory) > 4 else memory
    for msg in recent:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(HumanMessage(content=f"[助手之前说]: {msg['content'][:100]}"))

    messages.append(HumanMessage(content=message))

    result = llm.invoke(messages)
    return {"result": result.content.strip()}



def generate_answer(state: State) -> dict:
    message = state["message"]
    info = state["info"]
    memory = state["memory"]

    messages = [SystemMessage(content=f"根据提供的信息生成答案。信息：{info}")]

    recent = memory[-4:] if len(memory) > 4 else memory
    for msg in recent:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(HumanMessage(content=f"[助手之前说]: {msg['content'][:100]}"))

    messages.append(HumanMessage(content=message))

    result = llm.invoke(messages)
    return {"result": result.content.strip()}




def check_quality(state: State) -> dict:
    message = state["message"]
    result = llm.invoke([
        SystemMessage(content=(
            "检查答案是否符合问题。"
            f"问题：{message}"
            f"答案：{state['result']}"
            "如果答案符合问题，则返回True，否则返回False。"
        )),
        HumanMessage(content=message),
    ])
    quality = result.content.strip() == "True"
    return {"quality": quality, "rounds": state["rounds"]+ 1}


def route_after_intent(state: State) -> str:
    if state["intention"] == "company_policy":
        return "company_policy"       # 返回 key，不是节点名
    elif state["intention"] == "calculate":
        return "calculate"            # 返回 key，不是节点名
    else:
        return "chat"                 # 返回 key，不是节点名


graph = StateGraph(State)
graph.add_node("analyze_emotion", analyze_emotion)
graph.add_node("human_intervention", human_intervention)
graph.add_node("intent_recognition", intent_recognition)
graph.add_node("knowledge_retrieval", knowledge_retrieval)
graph.add_node("calculator", calculator)
graph.add_node("direct_answer", direct_answer)
graph.add_node("generate_answer", generate_answer)
graph.add_node("check_quality", check_quality)

graph.add_edge(START, "analyze_emotion")
graph.add_edge("knowledge_retrieval", "generate_answer")
graph.add_edge("calculator", "generate_answer")
graph.add_edge("generate_answer", "check_quality")
graph.add_edge("direct_answer", "check_quality")
graph.add_edge("human_intervention", END)

graph.add_conditional_edges(
    "analyze_emotion",
    route_after_emotion,
    {"to_human": "human_intervention", "to_intent": "intent_recognition"},
)


graph.add_conditional_edges(
    "intent_recognition",
    route_after_intent,
    {"company_policy": "knowledge_retrieval", "calculate": "calculator", "chat": "direct_answer"},
)
def route_after_quality(state: State) -> str:
    if state["quality"] or state["rounds"] >= 3:
        return "end"
    else:
        return "retry"

graph.add_conditional_edges(
    "check_quality",
    route_after_quality,
    {"end": END, "retry": "generate_answer"},
)

app = graph.compile()


# ============================================
# 测试运行
# ============================================

# def ask(message: str, history: list):
#     print(f"\n{'='*50}")
#     print(f"用户：{message}")
#     print(f"{'='*50}")
#     result = app.invoke({
#         "message": message,
#         "memory": history,
#         "emotion": "",
#         "need_human": False,
#         "intention": "",
#         "info": "",
#         "result": "",
#         "rounds": 0,
#         "quality": False,
#
#     })
#     # 追加本轮对话
#     history.append({"role": "user", "content": message})
#     history.append({"role": "assistant", "content": result["result"]})
#
#     print(f"\n助手：{result['result']}")
#     print(f"（情绪：{result['emotion']} | 意图：{result['intention']} | 轮次：{result['rounds']}）")
#     return result
#
#
# # 测试
# chat_history = []
# ask("公司的请假制度是什么？", chat_history)
# ask("那年假有几天？", chat_history)
# ask("病假呢？", chat_history)






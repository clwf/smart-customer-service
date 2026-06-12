"""
web.py
智能客服 Web 界面
"""

import gradio as gr
from app import app  # 从 app.py 导入已经构建好的图
from langchain_core.messages import HumanMessage


def chat(message: str, history: list):
    """Gradio 调用的对话函数"""

    # 把 Gradio 的 history 格式转成我们的格式
    chat_memory = []
    for msg in history:
        chat_memory.append({"role": msg["role"], "content": msg["content"]})

    # 调用 Agent
    result = app.invoke({
        "message": message,
        "memory": chat_memory,
        "emotion": "",
        "need_human": False,
        "intention": "",
        "info": "",
        "result": "",
        "rounds": 0,
        "quality": False,
    })

    return result["result"]


# 创建 Gradio 界面
demo = gr.ChatInterface(
    fn=chat,
    title="智能客服助手",
    description="支持公司制度问答 | 数学计算 | 日常聊天",
    examples=[
        "公司的请假制度是什么？",
        "帮我算一下 (5000+3000)*12",
        "年假有几天？",
    ],
)

demo.launch()

<div align='center'>
  <h1>智能客服系统</h1>
  <h3>基于 LangGraph 的智能客服解决方案</h3>
  <p><em>集成情感分析、意图识别、知识库问答和自动质检功能</em></p>
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/LangGraph-Agent-4CAF50?style=flat" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/DeepSeek-API-0066FF?style=flat" alt="DeepSeek"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat" alt="License"/>
</div>

---

## 🎯 项目介绍

&emsp;&emsp;客服工作重复性高、压力大？让 AI 帮你分担！

&emsp;&emsp;智能客服系统是一个基于 LangGraph 构建的端到端客服解决方案。系统能够自动分析用户情绪、识别用户意图、检索知识库生成答案，并进行质量检查。当用户情绪负面时，自动转接人工客服，确保服务质量。

## ✨ 功能特点

- 😊 **情感分析** - 自动识别用户情绪，负面情绪转人工
- 🎯 **意图识别** - 识别公司制度、计算、闲聊等意图
- 📚 **知识库问答** - 基于 RAG 回答公司相关问题
- 🔢 **数学计算** - 自动提取并计算数学表达式
- 🧠 **对话记忆** - 保持上下文连贯性
- ✅ **质量检查** - 自动验证答案质量，不满意自动重试（最多3次）

## 🔄 工作流程

```
用户输入
    ↓
情感分析 → 负面情绪 → 转人工客服
    ↓ 正面/中性
意图识别
    ├─ 公司制度 → 知识库检索 → 生成答案
    ├─ 计算 → 提取表达式 → 计算结果
    └─ 闲聊 → 直接回答
    ↓
质量检查 → 不通过 → 重新生成（最多3次）
    ↓ 通过
返回答案
```

## 📚 快速开始

### 1. 安装依赖

```bash
pip install langchain langchain-openai langchain-chroma langgraph numexpr python-dotenv
```

### 2. 配置 API 密钥

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_api_key_here
```

### 3. 添加知识库文档

将客服相关文档放入 `documents/` 目录。

### 4. 运行程序

```bash
python app.py
```

## 📖 使用示例

```python
from app import app

# 初始化对话历史
chat_history = []

# 第一轮对话
result = app.invoke({
    "message": "公司的年假有几天？",
    "memory": chat_history,
    "emotion": "",
    "need_human": False,
    "intention": "",
    "info": "",
    "result": "",
    "rounds": 0,
    "quality": False,
})
print(result["result"])

# 更新对话历史
chat_history.append({"role": "user", "content": "公司的年假有几天？"})
chat_history.append({"role": "assistant", "content": result["result"]})

# 第二轮对话（支持上下文）
result = app.invoke({
    "message": "那病假呢？",
    "memory": chat_history,
    ...
})
```

## 📁 项目结构

```
smart-customer-service/
├── app.py              # 主程序（LangGraph 工作流）
├── knowledge_base.py   # 知识库管理（RAG）
├── web.py              # Web 界面（可选）
├── documents/          # 知识库文档目录
├── chroma_db/          # 向量数据库
├── models/             # 模型相关文件
├── README.md           # 项目说明
└── .env                # API 密钥配置
```

## 🔧 技术栈

- **Python 3.8+** - 主要编程语言
- **LangChain** - LLM 应用开发框架
- **LangGraph** - 状态图编排框架
- **ChromaDB** - 向量数据库
- **DeepSeek API** - 大语言模型服务
- **numexpr** - 数学表达式计算

## 💡 如何学习

&emsp;&emsp;本项目是学习 LangGraph 状态图编程的绝佳案例。通过这个项目，你将学习到：

- 如何使用 StateGraph 定义工作流
- 如何实现条件路由和分支逻辑
- 如何实现循环和重试机制
- 如何设计复杂的状态管理

&emsp;&emsp;建议你先理解 `app.py` 中的 State 定义和各个节点函数，再理解图的构建和边的连接。

## 🤝 如何贡献

欢迎任何形式的贡献！

- 🐛 **报告 Bug** - 发现问题请提交 Issue
- 💡 **提出建议** - 有更好的想法欢迎讨论
- 📝 **完善功能** - 提交你的 Pull Request
- 📚 **丰富知识库** - 添加更多客服场景文档

## 📜 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。
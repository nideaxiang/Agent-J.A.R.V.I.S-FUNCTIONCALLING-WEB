原作者和其项目地址：https://github.com/chuyu2025/chu-star/tree/main/function_calling
感谢其分享的项目，我在其基础上进行了修改和个性化开发。

Inspired by J.A.R.V.I.S from Iron Man
一个具备实时搜索、数据分析、邮件发送等能力的智能助手。
<img width="1254" height="1285" alt="image" src="https://github.com/user-attachments/assets/4277ba2f-d32b-497c-9381-d7b71300cd3f" />


# 🧠 Jarvis AI Assistant

一个基于 **Flask + OpenAI Function Calling + Tavily Search** 构建的多功能智能助理系统。

> Inspired by J.A.R.V.I.S from *Iron Man*
> 一个具备实时搜索、数据分析、邮件发送等能力的智能助手。

---

## 🚀 项目简介

本项目实现了一个具备多工具调用能力的 AI 助手（Jarvis），支持：

* 💬 多轮对话
* ⏰ 查询当前时间
* 📧 发送邮件
* 📍 最近点查找与可视化
* 📊 数据统计分析
* 🌍 实时互联网搜索（基于 Tavily API）

系统采用 OpenAI Function Calling 架构，实现真正的“模型决定调用工具”。

---

## 🏗 系统架构

```text
User
  ↓
Flask API
  ↓
OpenAI Model (Function Calling)
  ↓
Tool Execution (Python)
  ↓
Return Tool Result
  ↓
Model Summarizes Final Response
  ↓
Frontend
```

核心思想：

* 模型负责决策是否调用工具
* 后端负责执行工具
* 工具执行结果再返回给模型进行总结
* 实现完整闭环

---

## 🛠 技术栈

* Python 3.9+
* Flask
* OpenAI API
* LangChain Community Tools
* Tavily Search API
* NumPy
* Matplotlib
* SMTP

---

## 📦 功能模块

### 1️⃣ 当前时间查询

通过 Function Calling 自动触发 `get_current_time`。

---

### 2️⃣ 邮件发送

支持：

* 发件人
* 收件人
* 主题
* 内容

调用 SMTP 完成邮件发送。

---

### 3️⃣ 最近点查找 + 可视化

输入目标点 `[x, y]`：

* 计算数据库中最近点
* 自动生成散点图
* 返回 Base64 图片

---

### 4️⃣ 数据分析

输入数值数组：

* 平均值
* 标准差
* 最大值
* 最小值
* 数据数量

---

### 5️⃣ 实时搜索（Tavily）

基于：

```python
from langchain_community.tools.tavily_search import TavilySearchResults
```

支持：

* 新闻
* 天气
* 科技动态
* 实时事件
* 人物近况

并通过 Function Calling + 二次模型生成，实现真正的自然语言总结。

---

## ⚙ 安装步骤

### 1️⃣ 克隆仓库

```bash
git clone https://github.com/your-username/jarvis-ai.git
cd jarvis-ai
```

---

### 2️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install flask openai numpy matplotlib langchain langchain-community tavily-python python-dotenv
```

---

### 3️⃣ 配置环境变量

创建 `.env` 文件：

```env
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
AUTHORIZATION_CODE=your_email_authorization_code
```

---

### 4️⃣ 运行项目

```bash
python app.py
```

访问：

```
http://127.0.0.1:5000
```

---

## 🧠 Function Calling 流程说明

本项目实现了完整的 Function Calling 闭环：

1. 模型返回 `tool_calls`
2. 后端执行工具
3. 将结果以 `tool` 角色加入对话
4. 再次调用模型生成最终自然语言回答

避免了常见问题：

> ❌ 只输出“正在查询…”
> ✅ 正确总结搜索结果

---

## 📂 项目结构

```text
.
├── app.py
├── templates/
│   └── index.html
├── num_data.json
├── .env
└── README.md
```

---

## 🔮 未来可扩展方向

* 长期记忆（向量数据库）
* 多工具链式执行
* LangChain Agent 化
* LangGraph 架构升级
* 语音输入输出
* 自动任务调度

---

## 🎯 适合人群

* 学习 Function Calling
* 学习 Agent 架构
* 构建 AI 工具系统
* 课程设计 / 毕业项目
* 个人作品展示

---

## 🧩 核心亮点

* 完整的 Function Calling 二次调用机制
* Tavily 实时搜索集成
* 工具模块化设计
* 清晰可扩展架构

---

## 📜 License

MIT License

---

## 🤖 作者

Daniel Duan

如果这个项目对您有帮助，请给个 ⭐。






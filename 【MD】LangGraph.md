# LangGraph 核心组件: Graphs, State, Nodes, Edges
## LangGraph
🦜🕸️LangGraph <font style="color:rgba(0, 0, 0, 0.87);">⚡</font><font style="color:rgba(0, 0, 0, 0.87);"> 以图的方式构建语言代理 </font><font style="color:rgba(0, 0, 0, 0.87);">⚡</font>

<font style="color:rgba(0, 0, 0, 0.87);">官方文档地址：</font>[https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)

LangGraph 是一个用于构建具有 LLMs 的有状态、多角色应用程序的库，用于创建代理和多代理工作流。与其他 LLM 框架相比，它提供了以下核心优势：循环、可控性和持久性。

LangGraph 允许您定义涉及循环的流程，这对于大多数代理架构至关重要。作为一种非常底层的框架，它提供了对应用程序的流程和状态的精细控制，这对创建可靠的代理至关重要。此外，LangGraph 包含内置的持久性，可以实现高级的“人机交互”和内存功能。

LangGraph 的灵感来自 Pregel 和 Apache Beam。公共接口借鉴了 NetworkX。

> Apache Beam 是一个开源的统一编程模型，用于构建复杂的数据处理管道。它支持批处理和流处理，能够跨多个大数据执行引擎无缝运行。本文将详细介绍 Apache Beam 的原理、基础使用、高级使用
>

> NetworkX<font style="color:rgb(51, 51, 51);">在02年5月产生，是用python语言编写的软件包，便于用户对复杂网络进行创建、操作和学习。利用networkx可以以标准化和非标准化的数据格式存储网络、生成多种随机网络和经典网络、分析网络结构、建立网络模型、设计新的网络算法、进行网络绘制等。</font>
>

LangGraph 由 LangChain Inc（LangChain 的创建者）构建，但可以在没有 LangChain 的情况下使用。

### <font style="color:rgba(0, 0, 0, 0.87);">主要功能</font>
+ **<font style="color:rgba(0, 0, 0, 0.87);">循环和分支</font>**<font style="color:rgba(0, 0, 0, 0.87);">：在您的应用程序中实现循环和条件语句。</font>
+ **<font style="color:rgba(0, 0, 0, 0.87);">持久性</font>**<font style="color:rgba(0, 0, 0, 0.87);">：在图中的每个步骤之后自动保存状态。在任何时候暂停和恢复图执行以支持错误恢复、“人机交互”工作流、时间旅行等等。</font>
+ **<font style="color:rgba(0, 0, 0, 0.87);">“人机交互”</font>**<font style="color:rgba(0, 0, 0, 0.87);">：中断图执行以批准或编辑代理计划的下一个动作。</font>
+ **<font style="color:rgba(0, 0, 0, 0.87);">流支持</font>**<font style="color:rgba(0, 0, 0, 0.87);">：在每个节点产生输出时流式传输输出（包括令牌流式传输）。</font>
+ **<font style="color:rgba(0, 0, 0, 0.87);">与 LangChain 集成</font>**<font style="color:rgba(0, 0, 0, 0.87);">：LangGraph 与</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">LangChain</font>](https://github.com/langchain-ai/langchain/)<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">和</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">LangSmith</font>](https://docs.smith.langchain.com/)<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">无缝集成（但不需要它们）。</font>

### <font style="color:rgba(0, 0, 0, 0.87);">安装</font>
```plain
pip install -U langgraph
```

### <font style="color:rgba(0, 0, 0, 0.87);">示例</font>
<font style="color:rgba(0, 0, 0, 0.87);">LangGraph 的一个核心概念是状态。每次图执行都会创建一个状态，该状态在图中的节点执行时传递，每个节点在执行后使用其返回值更新此内部状态。图更新其内部状态的方式由所选图类型或自定义函数定义。</font>

<font style="color:rgba(0, 0, 0, 0.87);">让我们看一个可以使用搜索工具的简单代理示例。</font>

```python
pip install langchain-openai
```

```plain
setx OPENAI_BASE_URL "https://api.openai.com/v1"
setx OPENAI_API_KEY "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

<font style="color:rgba(0, 0, 0, 0.87);">可以选择设置</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">LangSmith</font>](https://docs.smith.langchain.com/)<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">以实现最佳的可观察性。</font>

```plain
setx LANGSMITH_TRACING "true"
setx LANGSMITH_API_KEY "xxxxxxxxxxxxxxxx"
```

```python
#示例：langgraph_hello.py
from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
# pip install langgraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# 定义工具函数，用于代理调用外部工具
@tool
def search(query: str):
    """模拟一个搜索工具"""
    if "上海" in query.lower() or "Shanghai" in query.lower():
        return "现在30度，有雾."
    return "现在是35度，阳光明媚。"


# 将工具函数放入工具列表
tools = [search]

# 创建工具节点
tool_node = ToolNode(tools)

# 1.初始化模型和工具，定义并绑定工具到模型
model = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

# 定义函数，决定是否继续执行
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # 如果LLM调用了工具，则转到“tools”节点
    if last_message.tool_calls:
        return "tools"
    # 否则，停止（回复用户）
    return END


# 定义调用模型的函数
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    # 返回列表，因为这将被添加到现有列表中
    return {"messages": [response]}

# 2.用状态初始化图，定义一个新的状态图
workflow = StateGraph(MessagesState)
# 3.定义图节点，定义我们将循环的两个节点
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# 4.定义入口点和图边
# 设置入口点为“agent”
# 这意味着这是第一个被调用的节点
workflow.set_entry_point("agent")

# 添加条件边
workflow.add_conditional_edges(
    # 首先，定义起始节点。我们使用`agent`。
    # 这意味着这些边是在调用`agent`节点后采取的。
    "agent",
    # 接下来，传递决定下一个调用节点的函数。
    should_continue,
)

# 添加从`tools`到`agent`的普通边。
# 这意味着在调用`tools`后，接下来调用`agent`节点。
workflow.add_edge("tools", 'agent')

# 初始化内存以在图运行之间持久化状态
checkpointer = MemorySaver()

# 5.编译图
# 这将其编译成一个LangChain可运行对象，
# 这意味着你可以像使用其他可运行对象一样使用它。
# 注意，我们（可选地）在编译图时传递内存
app = workflow.compile(checkpointer=checkpointer)

# 6.执行图，使用可运行对象
final_state = app.invoke(
    {"messages": [HumanMessage(content="上海的天气怎么样?")]},
    config={"configurable": {"thread_id": 42}}
)
# 从 final_state 中获取最后一条消息的内容
result = final_state["messages"][-1].content
print(result)
final_state = app.invoke(
    {"messages": [HumanMessage(content="我问的那个城市?")]},
    config={"configurable": {"thread_id": 42}}
)
result = final_state["messages"][-1].content
print(result)
```

```plain
上海现在的天气是30度，有雾。
```

<font style="color:rgba(0, 0, 0, 0.87);">现在，当我们传递相同的</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`"thread_id"`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">时，对话上下文将通过保存的状态（即存储的消息列表）保留下来。</font>

```python
final_state = app.invoke(
    {"messages": [HumanMessage(content="我问的那个城市?")]},
    config={"configurable": {"thread_id": 42}}
)
result = final_state["messages"][-1].content
print(result)
```

```plain
你问的是上海的天气。上海现在的天气是30度，有雾。
```

### <font style="color:rgba(0, 0, 0, 0.87);">逐步分解</font>
#### 初始化模型和工具
    - 我们使用 `ChatOpenAI` 作为我们的 LLM。**注意：**我们需要确保模型知道可以使用哪些工具。我们可以通过将 LangChain 工具转换为 OpenAI 工具调用格式来完成此操作，方法是使用 `.bind_tools()` 方法。
    - 我们定义要使用的工具——在本例中是搜索工具。创建自己的工具非常容易——请参阅此处的文档了解如何操作 [此处](http://python.langchain.ac.cn/docs/modules/agents/tools/custom_tools)。

#### 用状态初始化图
    - 我们通过传递状态模式（在本例中为 `MessagesState`）来初始化图（`StateGraph`）。
    - `MessagesState` 是一个预构建的状态模式，它具有一个属性，一个 LangChain `Message` 对象列表，以及将每个节点的更新合并到状态中的逻辑。

#### 定义图节点
我们需要两个主要节点

    - `agent` 节点：负责决定采取什么（如果有）行动。
    - 调用工具的 `tools` 节点：如果代理决定采取行动，此节点将执行该行动。

#### 定义入口点和图边
首先，我们需要设置图执行的入口点——`agent` 节点。

然后，我们定义一个普通边和一个条件边。条件边意味着目的地取决于图状态（`MessageState`）的内容。在本例中，目的地在代理（LLM）决定之前是未知的。

    - 条件边：调用代理后，我们应该要么
        * a. 如果代理说要采取行动，则运行工具
        * b. 如果代理没有要求运行工具，则完成（回复用户）。
    - 普通边：调用工具后，图应该始终返回到代理以决定下一步操作。

#### 编译图
    - 当我们编译图时，我们将其转换为 LangChain Runnable，这会自动启用使用您的输入调用 `.invoke()`、`.stream()` 和 `.batch()`。
    - 我们还可以选择传递检查点对象以在图运行之间持久化状态，并启用内存、“人机交互”工作流、时间旅行等等。在本例中，我们使用 `MemorySaver`——一个简单的内存中检查点。

#### 执行图
    1. LangGraph 将输入消息添加到内部状态，然后将状态传递给入口点节点 `"agent"`。
    2. `"agent"` 节点执行，调用聊天模型。
    3. 聊天模型返回 `AIMessage`。LangGraph 将其添加到状态中。
    4. 图循环以下步骤，直到 `AIMessage` 上不再有 `tool_calls`。
        * 如果 `AIMessage` 具有 `tool_calls`，则 `"tools"` 节点执行。
        * `"agent"` 节点再次执行并返回 `AIMessage`。
    5. 执行进度到特殊的 `END` 值，并输出最终状态。因此，我们得到所有聊天消息的列表作为输出。



## <font style="color:rgba(0, 0, 0, 0.87);">Graph(图)</font>
<font style="color:rgba(0, 0, 0, 0.87);">LangGraph 的核心是将代理工作流建模为图。你可以使用三个关键组件来定义代理的行为</font>

1. [<font style="color:rgba(0, 0, 0, 0.87);">状态</font>](https://github.langchain.ac.cn/langgraph/concepts/low_level/#state)<font style="color:rgba(0, 0, 0, 0.87);">：一个共享的数据结构，表示应用程序的当前快照。它可以是任何 Python 类型，但通常是</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`TypedDict`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">或 Pydantic</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`BaseModel`<font style="color:rgba(0, 0, 0, 0.87);">。</font>
2. [<font style="color:rgba(0, 0, 0, 0.87);">节点</font>](https://github.langchain.ac.cn/langgraph/concepts/low_level/#nodes)<font style="color:rgba(0, 0, 0, 0.87);">：编码代理逻辑的 Python 函数。它们接收当前</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`状态`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">作为输入，执行一些计算或副作用，并返回一个更新的</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`状态`<font style="color:rgba(0, 0, 0, 0.87);">。</font>
3. [<font style="color:rgba(0, 0, 0, 0.87);">边</font>](https://github.langchain.ac.cn/langgraph/concepts/low_level/#edges)<font style="color:rgba(0, 0, 0, 0.87);">：Python 函数，根据当前</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`状态`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">确定要执行的下一个</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`节点`<font style="color:rgba(0, 0, 0, 0.87);">。它们可以是条件分支或固定转换。</font>

<font style="color:rgba(0, 0, 0, 0.87);">通过组合</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`节点`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">和</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`边`<font style="color:rgba(0, 0, 0, 0.87);">，你可以创建复杂的循环工作流，随着时间的推移发展</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`状态`<font style="color:rgba(0, 0, 0, 0.87);">。但是，真正的力量来自于 LangGraph 如何管理</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`状态`<font style="color:rgba(0, 0, 0, 0.87);">。需要强调的是：</font>`节点`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">和</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`边`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">不过是 Python 函数——它们可以包含 LLM 或简单的 Python 代码。</font>

<font style="color:rgba(0, 0, 0, 0.87);">简而言之：</font>_<font style="color:rgba(0, 0, 0, 0.87);">节点完成工作。边指示下一步要做什么</font>_<font style="color:rgba(0, 0, 0, 0.87);">。</font>

<font style="color:rgba(0, 0, 0, 0.87);">LangGraph 的底层图算法使用</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">消息传递</font>](https://en.wikipedia.org/wiki/Message_passing)<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">来定义一个通用程序。当一个节点完成其操作时，它会沿着一条或多条边向其他节点发送消息。这些接收节点然后执行其函数，将结果消息传递给下一组节点，并且该过程继续进行。受到 Google 的</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">Pregel</font>](https://research.google/pubs/pregel-a-system-for-large-scale-graph-processing/)<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">系统的启发，该程序以离散的“超级步骤”进行。</font>

<font style="color:rgba(0, 0, 0, 0.87);">超级步骤可以被认为是图节点上的单个迭代。并行运行的节点属于同一个超级步骤，而顺序运行的节点则属于不同的超级步骤。在图执行开始时，所有节点都处于</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`inactive`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">状态。当节点在任何传入边（或“通道”）上收到新消息（状态）时，它将变为</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`active`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">状态。然后，活动节点运行其函数并响应更新。在每个超级步骤结束时，没有传入消息的节点通过将其标记为</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`inactive`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">来投票</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`halt`<font style="color:rgba(0, 0, 0, 0.87);">。当所有节点都处于</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`inactive`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">状态且没有消息在传输中时，图执行终止。</font>

### <font style="color:rgba(0, 0, 0, 0.87);">StateGraph</font>
`StateGraph`<font style="color:rgba(0, 0, 0, 0.87);"> 类是使用的主要图类。它由用户定义的 </font>`状态`<font style="color:rgba(0, 0, 0, 0.87);"> 对象参数化。</font>

```python
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
class MyState(TypedDict)
    ...
graph = StateGraph(MyState)
```

<font style="color:rgba(0, 0, 0, 0.87);">基类：</font>`图`

<font style="color:rgba(0, 0, 0, 0.87);">一个图，其节点通过读取和写入共享状态进行通信。每个节点的签名是 State -> Partial</font><font style="color:rgba(0, 0, 0, 0.87);">.</font>

<font style="color:rgba(0, 0, 0, 0.87);">每个状态键可以选择性地使用一个 reducer 函数进行注释，该函数将用于聚合从多个节点接收到的该键的值。reducer 函数的签名是 (Value, Value) -> Value。</font>

**<font style="color:rgba(0, 0, 0, 0.87);">参数</font>**

+ `**state_schema**`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">(</font>`[类型](http://docs.pythonlang.cn/3/library/typing.html#typing.Type)[[任何](http://docs.pythonlang.cn/3/library/typing.html#typing.Any)]`<font style="color:rgba(0, 0, 0, 0.87);">, 默认值：</font>`None`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">) –</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>

<font style="color:rgba(0, 0, 0, 0.87);">定义状态的模式类。</font>

+ `**config_schema**`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">(</font>`[可选](http://docs.pythonlang.cn/3/library/typing.html#typing.Optional)[[类型](http://docs.pythonlang.cn/3/library/typing.html#typing.Type)[[任何](http://docs.pythonlang.cn/3/library/typing.html#typing.Any)]]`<font style="color:rgba(0, 0, 0, 0.87);">, 默认值：</font>`None`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">) –</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>

<font style="color:rgba(0, 0, 0, 0.87);">定义配置的模式类。使用此方法在您的 API 中公开可配置参数。</font>

**<font style="color:rgba(0, 0, 0, 0.87);">示例</font>**

```python
#示例：state_graph.py
# 从langgraph.graph模块导入START和StateGraph
from langgraph.graph import START, StateGraph

# 定义一个节点函数my_node，接收状态和配置，返回新的状态
def my_node(state, config):
    return {"x": state["x"] + 1,"y": state["y"] + 2}

# 创建一个状态图构建器builder，使用字典类型作为状态类型
builder = StateGraph(dict)
# 向构建器中添加节点my_node，节点名称将自动设置为'my_node'
builder.add_node(my_node)  # node name will be 'my_node'
# 添加一条边，从START到'my_node'节点
builder.add_edge(START, "my_node")
# 编译状态图，生成可执行的图
graph = builder.compile()
# 调用编译后的图，传入初始状态{"x": 1}
print(graph.invoke({"x": 1,"y":2}))

```

**结果**

```python
{'x': 2, 'y': 4}
```



### <font style="color:rgba(0, 0, 0, 0.87);">Compiling your graph(编译你的图)</font>
要构建你的图，你首先定义状态，然后添加节点和边，最后进行编译。编译图究竟是什么，为什么需要它？

<font style="color:rgba(0, 0, 0, 0.87);">编译是一个非常简单的步骤。它对图的结构进行一些基本检查（没有孤立的节点等等）。它也是你可以指定运行时参数的地方，例如</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">检查点</font>](https://github.langchain.ac.cn/langgraph/concepts/low_level/#checkpointer)<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">和</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">断点</font>](https://github.langchain.ac.cn/langgraph/concepts/low_level/#breakpoints)<font style="color:rgba(0, 0, 0, 0.87);">。你只需调用</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`.compile`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">方法即可编译你的图。</font>

```python
#你必须在使用图之前编译它。
graph = graph_builder.compile(...)
```

**编译结果**

```python
nodes={'__start__': PregelNode(config={'tags': ['langsmith:hidden'], 'metadata': {}, 'configurable': {}}, channels=['__start__'], triggers=['__start__'], writers=[ChannelWrite<__root__>(recurse=True, writes=[ChannelWriteEntry(channel='__root__', value=<object object at 0x00000180616FE0C0>, skip_none=True, mapper=None)], require_at_least_one_of=['__root__']), ChannelWrite<start:my_node>(recurse=True, writes=[ChannelWriteEntry(channel='start:my_node', value='__start__', skip_none=False, mapper=None)], require_at_least_one_of=None)]), 'my_node': PregelNode(config={'tags': [], 'metadata': {}, 'configurable': {}}, channels=['__root__'], triggers=['start:my_node'], writers=[ChannelWrite<my_node,__root__>(recurse=True, writes=[ChannelWriteEntry(channel='my_node', value='my_node', skip_none=False, mapper=None), ChannelWriteEntry(channel='__root__', value=<object object at 0x00000180616FE0C0>, skip_none=True, mapper=None)], require_at_least_one_of=['__root__'])])} channels={'__root__': <langgraph.channels.last_value.LastValue object at 0x0000018061C48470>, '__start__': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x0000018061C484A0>, 'my_node': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x0000018065131BB0>, 'start:my_node': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x0000018064DC0050>} auto_validate=False stream_mode='updates' output_channels='__root__' stream_channels='__root__' input_channels='__start__' builder=<langgraph.graph.state.StateGraph object at 0x0000018064DE0740>

```

<font style="color:rgba(0, 0, 0, 0.87);"></font>

## <font style="color:rgba(0, 0, 0, 0.87);">State(状态)</font>
<font style="color:rgba(0, 0, 0, 0.87);">定义图时，你做的第一件事是定义图的</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`状态`<font style="color:rgba(0, 0, 0, 0.87);">。</font>`状态`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">包含图的</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">模式</font>](https://github.langchain.ac.cn/langgraph/concepts/low_level/#schema)<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">以及</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">归约器函数</font>](https://github.langchain.ac.cn/langgraph/concepts/low_level/#reducers)<font style="color:rgba(0, 0, 0, 0.87);">，它们指定如何将更新应用于状态。</font>`状态`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">的模式将是图中所有</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`节点`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">和</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`边`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">的输入模式，可以是</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`TypedDict`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">或者</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`Pydantic`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">模型。所有</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`节点`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">将发出对</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`状态`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">的更新，这些更新然后使用指定的</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`归约器`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">函数进行应用。</font>

### <font style="color:rgba(0, 0, 0, 0.87);">Schema(模式)</font>
<font style="color:rgba(0, 0, 0, 0.87);">指定图模式的主要文档化方法是使用</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`TypedDict`<font style="color:rgba(0, 0, 0, 0.87);">。但是，我们也支持</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">使用 Pydantic BaseModel</font>](https://github.langchain.ac.cn/langgraph/how-tos/state-model/)<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">作为你的图状态，以添加</font>**<font style="color:rgba(0, 0, 0, 0.87);">默认值</font>**<font style="color:rgba(0, 0, 0, 0.87);">和其他数据验证。</font>

<font style="color:rgba(0, 0, 0, 0.87);">默认情况下，图将具有相同的输入和输出模式。如果你想更改这一点，你也可以直接指定显式输入和输出模式。当你有许多键，其中一些是显式用于输入，而另一些是用于输出时，这很有用。查看</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">此笔记本</font>](https://github.langchain.ac.cn/langgraph/how-tos/input_output_schema/)<font style="color:rgba(0, 0, 0, 0.87);">，了解如何使用。</font>

<font style="color:rgba(0, 0, 0, 0.87);">默认情况下，图中的所有节点都将共享相同的状态。这意味着它们将读取和写入相同的状态通道。可以在图中创建节点写入私有状态通道，用于内部节点通信——查看</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">此笔记本</font>](https://github.langchain.ac.cn/langgraph/how-tos/pass_private_state/)<font style="color:rgba(0, 0, 0, 0.87);">，了解如何执行此操作。</font>

### <font style="color:rgba(0, 0, 0, 0.87);">Reducers(归约器)</font>
<font style="color:rgba(0, 0, 0, 0.87);">归约器是理解节点更新如何应用于</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`状态`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">的关键。</font>`状态`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">中的每个键都有其自己的独立归约器函数。如果未显式指定归约器函数，则假设对该键的所有更新都应该覆盖它。存在几种不同类型的归约器，从默认类型的归约器开始</font>

#### <font style="color:rgba(0, 0, 0, 0.87);">Default Reducer(默认归约器)</font>
<font style="color:rgba(0, 0, 0, 0.87);">这两个示例展示了如何使用默认归约器</font>

```python
#示例：default_reducer.py
from typing import TypedDict, List, Dict, Any

class State(TypedDict):
    foo: int
    bar: List[str]

def update_state(current_state: State, updates: Dict[str, Any]) -> State:
    # 创建一个新的状态字典
    new_state = current_state.copy()
    # 更新状态字典中的值
    new_state.update(updates)
    return new_state

# 初始状态
state: State = {"foo": 1, "bar": ["hi"]}

# 第一个节点返回的更新
node1_update = {"foo": 2}
state = update_state(state, node1_update)
print(state)  # 输出: {'foo': 2, 'bar': ['hi']}

# 第二个节点返回的更新
node2_update = {"bar": ["bye"]}
state = update_state(state, node2_update)
print(state)  # 输出: {'foo': 2, 'bar': ['bye']}

```

<font style="color:rgba(0, 0, 0, 0.87);">在此示例中，没有为任何键指定归约器函数。假设图的输入是 </font>`{"foo": 1, "bar": ["hi"]}`<font style="color:rgba(0, 0, 0, 0.87);">。然后，假设第一个 </font>`节点`<font style="color:rgba(0, 0, 0, 0.87);"> 返回 </font>`{"foo": 2}`<font style="color:rgba(0, 0, 0, 0.87);">。这被视为对状态的更新。请注意，</font>`节点`<font style="color:rgba(0, 0, 0, 0.87);"> 不需要返回整个 </font>`状态`<font style="color:rgba(0, 0, 0, 0.87);"> 模式——只需更新即可。应用此更新后，</font>`状态`<font style="color:rgba(0, 0, 0, 0.87);"> 则变为 </font>`{"foo": 2, "bar": ["hi"]}`<font style="color:rgba(0, 0, 0, 0.87);">。如果第二个节点返回 </font>`{"bar": ["bye"]}`<font style="color:rgba(0, 0, 0, 0.87);">，则 </font>`状态`<font style="color:rgba(0, 0, 0, 0.87);"> 则变为 </font>`{"foo": 2, "bar": ["bye"]}`





## <font style="color:rgba(0, 0, 0, 0.87);">Nodes(节点)</font>
<font style="color:rgba(0, 0, 0, 0.87);">在 LangGraph 中，节点通常是 Python 函数（同步或</font>`async`<font style="color:rgba(0, 0, 0, 0.87);">），其中第一个位置参数是</font>[<font style="color:rgba(0, 0, 0, 0.87);">状态</font>](https://github.langchain.ac.cn/langgraph/concepts/low_level/#state)<font style="color:rgba(0, 0, 0, 0.87);">，（可选地），第二个位置参数是“配置”，包含可选的</font>[<font style="color:rgba(0, 0, 0, 0.87);">可配置参数</font>](https://github.langchain.ac.cn/langgraph/concepts/low_level/#configuration)<font style="color:rgba(0, 0, 0, 0.87);">（例如</font>`thread_id`<font style="color:rgba(0, 0, 0, 0.87);">）。</font>

<font style="color:rgba(0, 0, 0, 0.87);">类似于</font>`NetworkX`<font style="color:rgba(0, 0, 0, 0.87);">，您可以使用</font>[<font style="color:rgba(0, 0, 0, 0.87);">add_node</font>](https://github.langchain.ac.cn/langgraph/reference/graphs/#langgraph.graph.StateGraph.add_node)<font style="color:rgba(0, 0, 0, 0.87);">方法将这些节点添加到图形中</font>

```python
#示例：node_case.py
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START
from langgraph.graph import END

# 初始化 StateGraph，状态类型为字典
graph = StateGraph(dict)

# 定义节点
def my_node(state: dict, config: RunnableConfig):
    print("In node: ", config["configurable"]["user_id"])
    return {"results": f"Hello, {state['input']}!"}

def my_other_node(state: dict):
    return state

# 将节点添加到图中
graph.add_node("my_node", my_node)
graph.add_node("other_node", my_other_node)

# 连接节点以确保它们是可达的
graph.add_edge(START, "my_node")
graph.add_edge("my_node", "other_node")

graph.add_edge("other_node", END)

# 编译图
print(graph.compile())
```

<font style="color:rgba(0, 0, 0, 0.87);">在幕后，函数被转换为</font>[<font style="color:rgba(0, 0, 0, 0.87);">RunnableLambda</font>](http://python-api.langchain.ac.cn/en/latest/runnables/langchain_core.runnables.base.RunnableLambda.html#langchain_core.runnables.base.RunnableLambda)<font style="color:rgba(0, 0, 0, 0.87);">，它为您的函数添加了批处理和异步支持，以及本地跟踪和调试。</font>

<font style="color:rgba(0, 0, 0, 0.87);">如果您在没有指定名称的情况下将节点添加到图形中，它将被赋予一个默认名称，该名称等同于函数名称。</font>

```python
graph.add_node(my_node)
# You can then create edges to/from this node by referencing it as `"my_node"`
```

### `START`<font style="color:rgba(0, 0, 0, 0.87);">节点</font>
`START`<font style="color:rgba(0, 0, 0, 0.87);">节点是一个特殊节点，它代表将用户输入发送到图形的节点。引用此节点的主要目的是确定哪些节点应该首先被调用。</font>

```python
from langgraph.graph import START

graph.add_edge(START, "my_node")
graph.add_edge("my_node", "other_node")
```

### `END`<font style="color:rgba(0, 0, 0, 0.87);">节点</font>
`END`<font style="color:rgba(0, 0, 0, 0.87);">节点是一个特殊节点，它代表一个终端节点。当您想要指定哪些边在完成操作后没有动作时，会引用此节点。</font>

```python
from langgraph.graph import END

graph.add_edge("other_node", END)
```

## <font style="color:rgba(0, 0, 0, 0.87);">Edges(边)</font>
<font style="color:rgba(0, 0, 0, 0.87);">边定义了逻辑如何路由以及图形如何决定停止。这是您的代理如何工作以及不同节点如何相互通信的重要部分。有一些关键类型的边</font>

+ <font style="color:rgba(0, 0, 0, 0.87);">普通边：直接从一个节点到下一个节点。</font>
+ <font style="color:rgba(0, 0, 0, 0.87);">条件边：调用一个函数来确定下一个要转到的节点。</font>
+ <font style="color:rgba(0, 0, 0, 0.87);">入口点：用户输入到达时首先调用的节点。</font>
+ <font style="color:rgba(0, 0, 0, 0.87);">条件入口点：调用一个函数来确定用户输入到达时首先调用的节点。</font>

<font style="color:rgba(0, 0, 0, 0.87);">一个节点可以有多个输出边。如果一个节点有多个输出边，则所有这些目标节点将在下一个超级步骤中并行执行。</font>



### <font style="color:rgba(0, 0, 0, 0.87);">普通边</font>
<font style="color:rgba(0, 0, 0, 0.87);">如果您总是想从节点 A 到节点 B，您可以直接使用</font>[<font style="color:rgba(0, 0, 0, 0.87);">add_edge</font>](https://github.langchain.ac.cn/langgraph/reference/graphs/#langgraph.graph.StateGraph.add_edge)<font style="color:rgba(0, 0, 0, 0.87);">方法。</font>

```python
#示例：edges_case.py
graph.add_edge("node_a", "node_b")
```

### <font style="color:rgba(0, 0, 0, 0.87);">条件边</font>
<font style="color:rgba(0, 0, 0, 0.87);">如果您想选择性地路由到一个或多个边（或选择性地终止），您可以使用</font>[<font style="color:rgba(0, 0, 0, 0.87);">add_conditional_edges</font>](https://github.langchain.ac.cn/langgraph/reference/graphs/#langgraph.graph.StateGraph.add_conditional_edges)<font style="color:rgba(0, 0, 0, 0.87);">方法。此方法接受节点的名称和一个“路由函数”，该函数将在该节点执行后被调用</font>

```python
graph.add_conditional_edges("node_a", routing_function)
```

<font style="color:rgba(0, 0, 0, 0.87);">类似于节点，</font>`routing_function`<font style="color:rgba(0, 0, 0, 0.87);">接受图形的当前</font>`state`<font style="color:rgba(0, 0, 0, 0.87);">并返回一个值。</font>

<font style="color:rgba(0, 0, 0, 0.87);">默认情况下，返回值</font>`routing_function`<font style="color:rgba(0, 0, 0, 0.87);">用作要将状态发送到下一个节点的节点名称（或节点列表）。所有这些节点将在下一个超级步骤中并行运行。</font>

<font style="color:rgba(0, 0, 0, 0.87);">您可以选择提供一个字典，该字典将</font>`routing_function`<font style="color:rgba(0, 0, 0, 0.87);">的输出映射到下一个节点的名称。</font>

```python
graph.add_conditional_edges("node_a", routing_function, {True: "node_b", False: "node_c"})
```

### <font style="color:rgba(0, 0, 0, 0.87);">入口点</font>
<font style="color:rgba(0, 0, 0, 0.87);">入口点是图形启动时运行的第一个节点。您可以从虚拟的</font>[<font style="color:rgba(0, 0, 0, 0.87);">START</font>](https://github.langchain.ac.cn/langgraph/reference/graphs/#start)<font style="color:rgba(0, 0, 0, 0.87);">节点使用</font>[<font style="color:rgba(0, 0, 0, 0.87);">add_edge</font>](https://github.langchain.ac.cn/langgraph/reference/graphs/#langgraph.graph.StateGraph.add_edge)<font style="color:rgba(0, 0, 0, 0.87);">方法到要执行的第一个节点，以指定进入图形的位置。</font>

```python
from langgraph.graph import START

graph.add_edge(START, "my_node")
```

### <font style="color:rgba(0, 0, 0, 0.87);">条件入口点</font>
<font style="color:rgba(0, 0, 0, 0.87);">条件入口点允许您根据自定义逻辑从不同的节点开始。您可以从虚拟的</font>[<font style="color:rgba(0, 0, 0, 0.87);">START</font>](https://github.langchain.ac.cn/langgraph/reference/graphs/#start)<font style="color:rgba(0, 0, 0, 0.87);">节点使用</font>[<font style="color:rgba(0, 0, 0, 0.87);">add_conditional_edges</font>](https://github.langchain.ac.cn/langgraph/reference/graphs/#langgraph.graph.StateGraph.add_conditional_edges)<font style="color:rgba(0, 0, 0, 0.87);">来实现这一点。</font>

```python
from langgraph.graph import START

graph.add_conditional_edges(START, routing_function)
```

<font style="color:rgba(0, 0, 0, 0.87);">您可以选择提供一个字典，该字典将</font>`routing_function`<font style="color:rgba(0, 0, 0, 0.87);">的输出映射到下一个节点的名称。</font>

```python
graph.add_conditional_edges(START, routing_my,{True: "my_node", False: "other_node"})
```



# LangGraph 实现：持久化, Human-in-the-loop
## Persistence(持久化)
### 添加持久性内存
#### <font style="color:rgba(0, 0, 0, 0.87);">checkpointer(检查点)</font>
<font style="color:rgba(0, 0, 0, 0.87);">LangGraph 具有一个内置的持久化层，通过</font>[<font style="color:rgba(0, 0, 0, 0.87);">检查点</font>](https://github.langchain.ac.cn/langgraph/reference/checkpoints/#basecheckpointsaver)<font style="color:rgba(0, 0, 0, 0.87);">实现。当您将检查点与图形一起使用时，您可以与该图形的状态进行交互。当您将检查点与图形一起使用时，您可以与图形的状态进行交互并管理它。检查点在每个超级步骤中保存图形状态的检查点，从而实现一些强大的功能</font>

<font style="color:rgba(0, 0, 0, 0.87);">首先，检查点通过允许人类检查、中断和批准步骤来促进</font>[<font style="color:rgba(0, 0, 0, 0.87);">人机交互工作流</font>](https://github.langchain.ac.cn/langgraph/concepts/agentic_concepts/#human-in-the-loop)<font style="color:rgba(0, 0, 0, 0.87);">工作流。检查点对于这些工作流是必需的，因为人类必须能够在任何时候查看图形的状态，并且图形必须能够在人类对状态进行任何更新后恢复执行。</font>

<font style="color:rgba(0, 0, 0, 0.87);">其次，它允许在交互之间进行</font>[<font style="color:rgba(0, 0, 0, 0.87);">“记忆”</font>](https://github.langchain.ac.cn/langgraph/concepts/agentic_concepts/#memory)<font style="color:rgba(0, 0, 0, 0.87);">。您可以使用检查点创建线程并在图形执行后保存线程的状态。在重复的人类交互（例如对话）的情况下，任何后续消息都可以发送到该检查点，该检查点将保留对其以前消息的记忆。</font>

<font style="color:rgba(0, 0, 0, 0.87);"></font>

<font style="color:rgba(0, 0, 0, 0.87);">许多 AI 应用程序需要内存来跨多个交互共享上下文。在 LangGraph 中，通过</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">检查点</font>](https://github.com/langchain-ai/langgraph/tree/e4ca7ab69c599fd77dd4f0d47280849d715392cc/libs/checkpoint)<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">为任何</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">StateGraph</font>](https://github.langchain.ac.cn/langgraph/reference/graphs/#langgraph.graph.StateGraph)<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">提供内存。</font>

<font style="color:rgba(0, 0, 0, 0.87);">在创建任何 LangGraph 工作流时，您可以通过以下方式设置它们以持久保存其状态</font>

1. <font style="color:rgba(0, 0, 0, 0.87);">一个 </font>[<font style="color:rgba(0, 0, 0, 0.87);">检查点</font>](https://github.langchain.ac.cn/langgraph/reference/checkpoints/#basecheckpointsaver)<font style="color:rgba(0, 0, 0, 0.87);">，例如 MemorySaver</font>
2. <font style="color:rgba(0, 0, 0, 0.87);">在编译图时调用</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`compile(checkpointer=my_checkpointer)`<font style="color:rgba(0, 0, 0, 0.87);">。</font>

<font style="color:rgba(0, 0, 0, 0.87);">示例</font>

```python
#示例：persistence_case.py
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(....)
# ... define the graph
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
...
```

<font style="color:rgba(0, 0, 0, 0.87);">这适用于</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">StateGraph</font>](https://github.langchain.ac.cn/langgraph/reference/graphs/#langgraph.graph.StateGraph)<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">及其所有子类，例如</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">MessageGraph</font>](https://github.langchain.ac.cn/langgraph/reference/graphs/#messagegraph)<font style="color:rgba(0, 0, 0, 0.87);">。</font>

<font style="color:rgba(0, 0, 0, 0.87);">以下是一个示例。</font>

**注意**

在本操作指南中，我们将从头开始创建我们的代理，以保持透明度（但冗长）。您可以使用 `create_react_agent(model, tools=tool, checkpointer=checkpointer)` ([API 文档](https://github.langchain.ac.cn/langgraph/reference/prebuilt/#create_react_agent)) 构造函数完成类似的功能。如果您习惯使用 LangChain 的 [AgentExecutor](http://python.langchain.ac.cn/v0.2/docs/how_to/agent_executor/#concepts) 类，这可能更合适。

#### <font style="color:rgba(0, 0, 0, 0.87);">设置</font>
<font style="color:rgba(0, 0, 0, 0.87);">首先，我们需要安装所需的软件包</font>

```plain
%pip install --quiet -U langgraph langchain_openai
```

<font style="color:rgba(0, 0, 0, 0.87);">接下来，我们需要设置 OpenAI（我们将使用的 LLM）和 Tavily（我们将使用的搜索工具）的 API 密钥</font>

<font style="color:rgba(0, 0, 0, 0.87);">可选地，我们可以设置 </font>[<font style="color:rgba(0, 0, 0, 0.87);">LangSmith 跟踪</font>](https://smith.langchain.com/)<font style="color:rgba(0, 0, 0, 0.87);"> 的 API 密钥，这将为我们提供一流的可观察性。</font>

#### <font style="color:rgba(0, 0, 0, 0.87);">设置状态</font>
<font style="color:rgba(0, 0, 0, 0.87);">状态是所有节点的接口。</font>

```python
# 导入所需的类型注解和模块
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

# 定义一个状态类，包含一个消息列表，消息列表带有 add_messages 注解
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

#### <font style="color:rgba(0, 0, 0, 0.87);">设置工具</font>
<font style="color:rgba(0, 0, 0, 0.87);">我们首先将定义要使用的工具。对于这个简单的示例，我们将创建一个占位符搜索引擎。但是，创建自己的工具非常容易 - 请参阅</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">此处</font>](http://python.langchain.ac.cn/v0.2/docs/how_to/custom_tools)<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">的文档了解如何操作。</font>

```python
# 从 langchain_core.tools 导入工具装饰器
from langchain_core.tools import tool


# 定义一个名为 search 的工具函数，用于模拟网络搜索
@tool
def search(query: str):
    """Call to surf the web."""
    # 这是实际实现的占位符
    return ["The answer to your question lies within."]


# 将工具函数存入列表
tools = [search]
```

<font style="color:rgba(0, 0, 0, 0.87);">现在我们可以创建我们的</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">ToolNode</font>](https://github.langchain.ac.cn/langgraph/reference/prebuilt/?h=tool+node#toolnode)<font style="color:rgba(0, 0, 0, 0.87);">。此对象实际上</font>**<font style="color:rgba(0, 0, 0, 0.87);">运行</font>**<font style="color:rgba(0, 0, 0, 0.87);">LLM 要求使用的工具（即函数）。</font>

```python
from langgraph.prebuilt import ToolNode

# 创建一个 ToolNode 实例，传入工具列表
tool_node = ToolNode(tools)
```

#### <font style="color:rgba(0, 0, 0, 0.87);">设置模型</font>
<font style="color:rgba(0, 0, 0, 0.87);">现在我们需要加载</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">聊天模型</font>](http://python.langchain.ac.cn/v0.2/docs/concepts/#chat-models)<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">来为我们的代理提供动力。对于以下设计，它必须满足两个条件</font>

1. <font style="color:rgba(0, 0, 0, 0.87);">它应该与</font>**<font style="color:rgba(0, 0, 0, 0.87);">消息</font>**<font style="color:rgba(0, 0, 0, 0.87);">一起使用（因为我们的状态包含聊天消息列表）</font>
2. <font style="color:rgba(0, 0, 0, 0.87);">它应该与</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">工具调用</font>](http://python.langchain.ac.cn/v0.2/docs/concepts/#functiontool-calling)<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">一起使用。</font>

**注意**

这些模型要求不是使用 LangGraph 的一般要求 - 它们只是此特定示例的要求。

```python
# 从 langchain_openai 导入 ChatOpenAI 模型
from langchain_openai import ChatOpenAI

# 创建一个 ChatOpenAI 模型实例，设置 streaming=True 以便可以流式传输 tokens
model = ChatOpenAI(temperature=0, streaming=True)
```

<font style="color:rgba(0, 0, 0, 0.87);">完成此操作后，我们应该确保模型知道它可以使用这些工具。我们可以通过将 LangChain 工具转换为 OpenAI 函数调用格式，然后将其绑定到模型类来实现这一点。</font>

```python
# 将工具绑定到模型上
bound_model = model.bind_tools(tools)
```

#### <font style="color:rgba(0, 0, 0, 0.87);">定义图</font>
<font style="color:rgba(0, 0, 0, 0.87);">现在我们需要在我们的图中定义几个不同的节点。在</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`langgraph`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">中，节点可以是函数或</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">可运行的</font>](http://python.langchain.ac.cn/v0.2/docs/concepts/#langchain-expression-language-lcel)<font style="color:rgba(0, 0, 0, 0.87);">。我们需要为此定义两个主要节点</font>

1. <font style="color:rgba(0, 0, 0, 0.87);">代理：负责决定要采取哪些（如果有）操作。</font>
2. <font style="color:rgba(0, 0, 0, 0.87);">一个用于调用工具的函数：如果代理决定采取操作，则此节点将执行该操作。</font>

<font style="color:rgba(0, 0, 0, 0.87);">我们还需要定义一些边。其中一些边可能是条件的。它们是条件的原因是，基于节点的输出，可能会采用几种路径之一。在运行该节点之前，无法知道要采用哪条路径（LLM 决定）。</font>

1. <font style="color:rgba(0, 0, 0, 0.87);">条件边：在调用代理后，我们应该：a. 如果代理说要采取操作，则应调用调用工具的函数 b. 如果代理说它已完成，则应完成</font>
2. <font style="color:rgba(0, 0, 0, 0.87);">普通边：在调用工具后，它应该始终返回到代理以决定下一步要做什么</font>

<font style="color:rgba(0, 0, 0, 0.87);">让我们定义节点，以及一个函数来决定如何采取哪些条件边。</font>

```python
# 导入 Literal 类型
from typing import Literal


# 定义一个函数，根据状态决定是否继续执行
def should_continue(state: State) -> Literal["action", "__end__"]:
    """Return the next node to execute."""
    last_message = state["messages"][-1]
    # 如果没有函数调用，则结束
    if not last_message.tool_calls:
        return "__end__"
    # 否则继续执行
    return "action"


# 定义一个函数调用模型
def call_model(state: State):
    response = model.invoke(state["messages"])
    # 返回一个列表，因为这将被添加到现有列表中
    return {"messages": response}
```

<font style="color:rgba(0, 0, 0, 0.87);">现在我们可以将所有内容放在一起并定义图！</font>

```python
# 从 langgraph.graph 导入 StateGraph 和 START
from langgraph.graph import StateGraph, START

# 定义一个新的图形工作流
workflow = StateGraph(State)

# 添加两个节点，分别是 agent 和 action
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# 设置入口点为 agent
workflow.add_edge(START, "agent")

# 添加条件边，根据 should_continue 函数决定下一个节点
workflow.add_conditional_edges(
    "agent",
    should_continue,
)

# 添加从 action 到 agent 的普通边
workflow.add_edge("action", "agent")
```

**<font style="color:rgba(0, 0, 0, 0.87);">持久性</font>**

<font style="color:rgba(0, 0, 0, 0.87);">要添加持久性，我们在编译图时传入一个检查点</font>

```python
# 从 langgraph.checkpoint.memory 导入 MemorySaver
from langgraph.checkpoint.memory import MemorySaver

# 创建一个 MemorySaver 实例
memory = MemorySaver()
```

```python
# 编译工作流，生成一个 LangChain Runnable
app = workflow.compile(checkpointer=memory)
```

**注意**

如果您使用的是 LangGraph Cloud，则**无需**在编译图时传递检查点，因为它会自动完成。

```plain
# 将生成的图片保存到文件
graph_png = app.get_graph().draw_mermaid_png()
with open("persistence_case.png", "wb") as f:
    f.write(graph_png)
```

![](https://cdn.nlark.com/yuque/0/2024/jpeg/2424104/1723622700579-6d230a39-b7b9-4990-962b-93b7a3dbd843.jpeg)

#### <font style="color:rgba(0, 0, 0, 0.87);">与代理交互</font>
<font style="color:rgba(0, 0, 0, 0.87);">现在我们可以与代理进行交互，并看到它会记住以前的消息！</font>

```python
# 从 langchain_core.messages 导入 HumanMessage
from langchain_core.messages import HumanMessage

# 设置配置参数
config = {"configurable": {"thread_id": "2"}}

# 创建一个 HumanMessage 实例，内容为 "hi! I'm bob"
input_message = HumanMessage(content="hi! I'm bob")

# 在流模式下运行应用程序，传入消息和配置，逐个打印每个事件的最后一条消息
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
```

```python
================================ Human Message =================================

hi! I'm bob
================================== Ai Message ==================================

Hello Bob! How can I assist you today?
```

```python
# 创建一个 HumanMessage 实例，内容为 "what is my name?"
input_message = HumanMessage(content="what is my name?")

# 在流模式下运行应用程序，传入消息和配置，逐个打印每个事件的最后一条消息
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
```

```plain
================================ Human Message =================================

what is my name?
================================== Ai Message ==================================

Your name is Bob.
```

<font style="color:rgba(0, 0, 0, 0.87);">如果我们想开始新的对话，可以传入不同的线程 ID。瞧！所有的记忆都消失了！</font>

```python
# 创建一个 HumanMessage 实例，内容为 "what is my name?"
input_message = HumanMessage(content="what is my name?")

# 在流模式下运行应用程序，传入消息和新的配置，逐个打印每个事件的最后一条消息
for event in app.stream(
    {"messages": [input_message]},
    {"configurable": {"thread_id": "3"}},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
```

```plain
================================ Human Message =================================

what is my name?
================================== Ai Message ==================================

I'm sorry, I do not know your name as I am an AI assistant and do not have access to personal information.
```

<font style="color:rgba(0, 0, 0, 0.87);">所有检查点都将持久保存到检查点，因此您可以随时恢复以前的线程。</font>

```python
# 创建一个 HumanMessage 实例，内容为 "You forgot??"
input_message = HumanMessage(content="You forgot??")

# 在流模式下运行应用程序，传入消息和原来的配置，逐个打印每个事件的最后一条消息
for event in app.stream(
    {"messages": [input_message]},
    {"configurable": {"thread_id": "2"}},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
```

```python
================================ Human Message =================================

You forgot??
================================== Ai Message ==================================

I apologize for the confusion. I am an AI assistant and I do not have the ability to remember information from previous interactions. How can I assist you today, Bob?
```

## Human-in-the-loop(人机交互)
### 添加断点
<font style="color:rgba(0, 0, 0, 0.87);">在某些节点执行之前或之后设置断点通常很有用。这可以用来在继续之前等待人工批准。当您</font>[<font style="color:rgba(0, 0, 0, 0.87);">“编译”图形</font>](https://github.langchain.ac.cn/langgraph/concepts/low_level/#compiling-your-graph)<font style="color:rgba(0, 0, 0, 0.87);">时，可以设置这些断点。您可以在节点执行</font>_<font style="color:rgba(0, 0, 0, 0.87);">之前</font>_<font style="color:rgba(0, 0, 0, 0.87);">（使用</font>`interrupt_before`<font style="color:rgba(0, 0, 0, 0.87);">）或节点执行</font>_<font style="color:rgba(0, 0, 0, 0.87);">之后</font>_<font style="color:rgba(0, 0, 0, 0.87);">（使用</font>`interrupt_after`<font style="color:rgba(0, 0, 0, 0.87);">）设置断点。</font>

<font style="color:rgba(0, 0, 0, 0.87);">使用断点时，您</font>**<font style="color:rgba(0, 0, 0, 0.87);">必须</font>**<font style="color:rgba(0, 0, 0, 0.87);">使用</font>[<font style="color:rgba(0, 0, 0, 0.87);">检查点</font>](https://github.langchain.ac.cn/langgraph/concepts/low_level/#checkpointer)<font style="color:rgba(0, 0, 0, 0.87);">。这是因为您的图形需要能够恢复执行。</font>

<font style="color:rgba(0, 0, 0, 0.87);">为了恢复执行，您可以使用</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`None`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">作为输入调用您的图。</font>

```python
# Initial run of graph
graph.invoke(inputs, config=config)

# Let's assume it hit a breakpoint somewhere, you can then resume by passing in None
graph.invoke(None, config=config)
```

<font style="color:rgba(0, 0, 0, 0.87);">有关如何添加断点的完整演练，请参阅</font>[<font style="color:rgba(0, 0, 0, 0.87);">本指南</font>](https://github.langchain.ac.cn/langgraph/how-tos/human_in_the_loop/breakpoints/)<font style="color:rgba(0, 0, 0, 0.87);">。</font>

<font style="color:rgba(0, 0, 0, 0.87);">人机交互 (HIL) 在</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">代理系统</font>](https://github.langchain.ac.cn/langgraph/concepts/agentic_concepts/#human-in-the-loop)<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">中至关重要。</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">断点</font>](https://github.langchain.ac.cn/langgraph/concepts/low_level/#breakpoints)<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">是一种常见的 HIL 交互模式，允许图在特定步骤停止并寻求人为批准后再继续执行（例如，对于敏感操作）。</font>

<font style="color:rgba(0, 0, 0, 0.87);">断点建立在 LangGraph </font>[<font style="color:rgba(0, 0, 0, 0.87);">检查点</font>](https://github.langchain.ac.cn/langgraph/concepts/low_level/#checkpointer)<font style="color:rgba(0, 0, 0, 0.87);"> 之上，检查点在每个节点执行后保存图的状态。 检查点保存在 </font>[<font style="color:rgba(0, 0, 0, 0.87);">线程</font>](https://github.langchain.ac.cn/langgraph/concepts/low_level/#threads)<font style="color:rgba(0, 0, 0, 0.87);"> 中，这些线程保存图状态，并且可以在图执行完成后访问。 这使得图执行可以在特定点暂停，等待人为批准，然后从最后一个检查点恢复执行。</font>

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1723887450308-e3f14fcd-4275-49d4-bd99-a82327acd88a.png)



<font style="color:rgba(0, 0, 0, 0.87);">让我们看看它的基本用法。</font>

<font style="color:rgba(0, 0, 0, 0.87);">下面，我们做两件事</font>

1. <font style="color:rgba(0, 0, 0, 0.87);">我们使用</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`interrupt_before`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">指定步骤，来指定</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">断点</font>](https://github.langchain.ac.cn/langgraph/concepts/low_level/#breakpoints)<font style="color:rgba(0, 0, 0, 0.87);">。</font>
2. <font style="color:rgba(0, 0, 0, 0.87);">我们设置一个</font>[<font style="color:rgba(0, 0, 0, 0.87);">检查点</font>](https://github.langchain.ac.cn/langgraph/concepts/low_level/#checkpointer)<font style="color:rgba(0, 0, 0, 0.87);">来保存图的状态。</font>

```python
#示例：breakpoints_case.py
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display


class State(TypedDict):
    input: str


def step_1(state):
    print("---Step 1---")
    pass


def step_2(state):
    print("---Step 2---")
    pass


def step_3(state):
    print("---Step 3---")
    pass


builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

# Set up memory
memory = MemorySaver()

# Add
graph = builder.compile(checkpointer=memory, interrupt_before=["step_3"])

# 将生成的图片保存到文件
graph_png = graph.get_graph().draw_mermaid_png()
with open("breakpoints_case.png", "wb") as f:
    f.write(graph_png)
```

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1723950093083-1f5e87be-5080-45b0-9044-1bfc497cd86e.png)

<font style="color:rgba(0, 0, 0, 0.87);">我们为检查点创建一个 </font>[<font style="color:rgba(0, 0, 0, 0.87);">线程 ID</font>](https://github.langchain.ac.cn/langgraph/concepts/low_level/#threads)<font style="color:rgba(0, 0, 0, 0.87);">。</font>

<font style="color:rgba(0, 0, 0, 0.87);">我们运行到步骤 3，如</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>`interrupt_before`<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">中定义。</font>

<font style="color:rgba(0, 0, 0, 0.87);">在用户输入/批准后，</font>[<font style="color:rgba(0, 0, 0, 0.87);">我们恢复执行</font>](https://github.langchain.ac.cn/langgraph/concepts/low_level/#breakpoints)<font style="color:rgba(0, 0, 0, 0.87);">，方法是用 </font>`None`<font style="color:rgba(0, 0, 0, 0.87);"> 调用图。</font>

```python
#示例：breakpoints_add.py
from typing import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    input: str

def step_1(state):
    print("---Step 1---")
    pass

def step_2(state):
    print("---Step 2---")
    pass

def step_3(state):
    print("---Step 3---")
    pass

builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

# Set up memory
memory = MemorySaver()

# Add
graph = builder.compile(checkpointer=memory, interrupt_before=["step_3"])

# Input
initial_input = {"input": "hello world"}

# Thread
thread = {"configurable": {"thread_id": "1"}}

# 运行graph，直到第一次中断
for event in graph.stream(initial_input, thread, stream_mode="values"):
    print(event)

user_approval = input("Do you want to go to Step 3? (yes/no): ")

if user_approval.lower() == "yes":
    # If approved, continue the graph execution
    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)
else:
    print("Operation cancelled by user.")
```

```python
{'input': 'hello world'}
---Step 1---
    ---Step 2---
```



# Agent 使用案例: Multi-Agent Systems, Planning Agent
## Multi-Agent Systems(多代理系统)
### 协作
<font style="color:rgba(0, 0, 0, 0.87);">单个代理通常可以使用少量工具在一个域内有效地运行，但即使使用像</font>`gpt-4`<font style="color:rgba(0, 0, 0, 0.87);">这样的强大模型，它在使用许多工具时也可能效率较低。</font>

<font style="color:rgba(0, 0, 0, 0.87);">解决复杂任务的一种方法是使用“分而治之”的方法：为每个任务或域创建一个专门的代理，并将任务路由到正确的“专家”。</font>

<font style="color:rgba(0, 0, 0, 0.87);">此笔记本（受 Wu 等人撰写的论文</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">AutoGen：通过多代理对话实现下一代 LLM 应用</font>](https://arxiv.org/abs/2308.08155)<font style="color:rgba(0, 0, 0, 0.87);"> </font><font style="color:rgba(0, 0, 0, 0.87);">的启发）展示了一种使用 LangGraph 进行此操作的方法。</font>

<font style="color:rgba(0, 0, 0, 0.87);">生成的图将类似于以下图</font>

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1724054341713-a29d9357-8198-48f1-9d5e-fd0cfc28121b.png)

<font style="color:rgba(0, 0, 0, 0.87);">在我们开始之前，快速说明一下：以下展示</font>_<font style="color:rgba(0, 0, 0, 0.87);">如何</font>_<font style="color:rgba(0, 0, 0, 0.87);">在 LangGraph 中实现某些设计模式。如果模式适合您的需求</font>

```python
%pip install -U langchain langchain_openai langsmith pandas langchain_experimental matplotlib langgraph langchain_core
```

```python
setx TAVILY_API_KEY ""
# Optional, add tracing in LangSmith
setx LANGCHAIN_TRACING_V2 "true"
setx LANGCHAIN_API_KEY ""
```

#### <font style="color:rgba(0, 0, 0, 0.87);">创建代理</font>
<font style="color:rgba(0, 0, 0, 0.87);">以下辅助函数将帮助创建代理。这些代理将成为图中的节点。</font>

<font style="color:rgba(0, 0, 0, 0.87);">如果您只想查看图的外观，可以跳过此步骤。</font>

```python
#示例：collaboration.py
# 导入基本消息类、用户消息类和工具消息类
from langchain_core.messages import (
BaseMessage,
HumanMessage,
ToolMessage,
)
# 导入聊天提示模板和消息占位符
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# 导入状态图相关的常量和类
from langgraph.graph import END, StateGraph, START


# 定义一个函数，用于创建代理
def create_agent(llm, tools, system_message: str):
    """创建一个代理。"""
    # 创建一个聊天提示模板
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个有帮助的AI助手，与其他助手合作。"
                " 使用提供的工具来推进问题的回答。"
                " 如果你不能完全回答，没关系，另一个拥有不同工具的助手"
                " 会接着你的位置继续帮助。执行你能做的以取得进展。"
                " 如果你或其他助手有最终答案或交付物，"
                " 在你的回答前加上FINAL ANSWER，以便团队知道停止。"
                " 你可以使用以下工具: {tool_names}。\n{system_message}",
            ),
            # 消息占位符
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    # 传递系统消息参数
    prompt = prompt.partial(system_message=system_message)
    # 传递工具名称参数
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    # 绑定工具并返回提示模板
    return prompt | llm.bind_tools(tools)
```

#### <font style="color:rgba(0, 0, 0, 0.87);">定义工具</font>
<font style="color:rgba(0, 0, 0, 0.87);">我们还将定义一些代理将在未来使用的工具</font>

```python
# 导入注解类型
from typing import Annotated

# 导入Tavily搜索工具
from langchain_community.tools.tavily_search import TavilySearchResults
# 导入工具装饰器
from langchain_core.tools import tool
# 导入Python REPL工具
from langchain_experimental.utilities import PythonREPL

# 创建Tavily搜索工具实例，设置最大结果数为5
tavily_tool = TavilySearchResults(max_results=5)

# 警告：这会在本地执行代码，未沙箱化时可能不安全
# 创建Python REPL实例
repl = PythonREPL()


# 定义一个工具函数，用于执行Python代码
@tool
def python_repl(
    code: Annotated[str, "要执行以生成图表的Python代码。"],
):
    """使用这个工具来执行Python代码。如果你想查看某个值的输出，
    应该使用print(...)。这个输出对用户可见。"""
    try:
        # 尝试执行代码
        result = repl.run(code)
    except BaseException as e:
        # 捕捉异常并返回错误信息
        return f"执行失败。错误: {repr(e)}"
    # 返回执行结果
    result_str = f"成功执行:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\n如果你已完成所有任务，请回复FINAL ANSWER。"
    )
```

#### <font style="color:rgba(0, 0, 0, 0.87);">创建图</font>
<font style="color:rgba(0, 0, 0, 0.87);">现在我们已经定义了工具并创建了一些辅助函数，将在下面创建各个代理，并告诉他们如何使用 LangGraph 相互交流。</font>

##### <font style="color:rgba(0, 0, 0, 0.87);">定义状态</font>
<font style="color:rgba(0, 0, 0, 0.87);">我们首先定义图的状态。这只是一个消息列表，以及一个用于跟踪最新发送者的键</font>

```python
# 导入操作符和类型注解
import operator
from typing import Annotated, Sequence, TypedDict

# 导入OpenAI聊天模型
from langchain_openai import ChatOpenAI


# 定义一个对象，用于在图的每个节点之间传递
# 我们将为每个代理和工具创建不同的节点
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
```

##### <font style="color:rgba(0, 0, 0, 0.87);">定义代理节点</font>
<font style="color:rgba(0, 0, 0, 0.87);">现在我们需要定义节点。首先，让我们定义代理的节点。</font>

```python
# 导入functools模块
import functools

# 导入AI消息类
from langchain_core.messages import AIMessage


# 辅助函数，用于为给定的代理创建节点
def agent_node(state, agent, name):
    # 调用代理
    result = agent.invoke(state)
    # 将代理输出转换为适合附加到全局状态的格式
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # 由于我们有一个严格的工作流程，我们可以
        # 跟踪发送者，以便知道下一个传递给谁。
        "sender": name,
    }


# 创建OpenAI聊天模型实例
llm = ChatOpenAI(model="gpt-4o")

# 研究代理和节点
research_agent = create_agent(
    llm,
    [tavily_tool],
    system_message="你应该提供准确的数据供chart_generator使用。",
)
# 创建研究节点
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# 图表生成器
chart_agent = create_agent(
    llm,
    [python_repl],
    system_message="你展示的任何图表都将对用户可见。",
)
# 创建图表生成节点
chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")
```

##### <font style="color:rgba(0, 0, 0, 0.87);">定义工具节点</font>
<font style="color:rgba(0, 0, 0, 0.87);">我们现在定义一个节点来运行工具</font>

```python
# 导入预构建的工具节点
from langgraph.prebuilt import ToolNode

# 定义工具列表
tools = [tavily_tool, python_repl]
# 创建工具节点
tool_node = ToolNode(tools)
```

##### <font style="color:rgba(0, 0, 0, 0.87);">定义边逻辑</font>
<font style="color:rgba(0, 0, 0, 0.87);">我们可以定义一些边缘逻辑，这些逻辑需要根据代理的结果来决定要执行的操作</font>

```python
# 任一代理都可以决定结束
from typing import Literal

# 定义路由器函数
def router(state) -> Literal["call_tool", "__end__", "continue"]:
    # 这是路由器
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # 上一个代理正在调用工具
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # 任何代理决定工作完成
        return "__end__"
    return "continue"
```

##### <font style="color:rgba(0, 0, 0, 0.87);">定义图</font>
<font style="color:rgba(0, 0, 0, 0.87);">我们现在可以将所有内容整合在一起，并定义图！</font>

```python
# 创建状态图实例
workflow = StateGraph(AgentState)

# 添加研究员节点
workflow.add_node("Researcher", research_node)
# 添加图表生成器节点
workflow.add_node("chart_generator", chart_node)
# 添加工具调用节点
workflow.add_node("call_tool", tool_node)

# 添加条件边
workflow.add_conditional_edges(
    "Researcher",
    router,
    {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END},
)
workflow.add_conditional_edges(
    "chart_generator",
    router,
    {"continue": "Researcher", "call_tool": "call_tool", "__end__": END},
)

# 添加条件边
workflow.add_conditional_edges(
    "call_tool",
    # 每个代理节点更新'sender'字段
    # 工具调用节点不更新，这意味着
    # 该边将路由回调用工具的原始代理
    lambda x: x["sender"],
    {
        "Researcher": "Researcher",
        "chart_generator": "chart_generator",
    },
)
# 添加起始边
workflow.add_edge(START, "Researcher")
# 编译工作流图
graph = workflow.compile()
```

```python
# 将生成的图片保存到文件
graph_png = graph.get_graph().draw_mermaid_png()
with open("collaboration.png", "wb") as f:
    f.write(graph_png)
```

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1724582598011-9a7c5436-7db1-4138-acf6-f2b6bb05b4fa.png)

#### <font style="color:rgba(0, 0, 0, 0.87);">调用</font>
<font style="color:rgba(0, 0, 0, 0.87);">图创建完毕后，您可以调用它！让我们让它为我们绘制一些统计数据。</font>

```python
# 事件流
events = graph.stream(
    {
        "messages": [
            HumanMessage(
                content="获取过去5年AI软件市场规模，"
                        " 然后绘制一条折线图。"
                        " 一旦你编写好代码，完成任务。"
            )
        ],
    },
    # 图中最多执行的步骤数
    {"recursion_limit": 150},
)
# 打印事件流中的每个状态
for s in events:
    print(s)
    print("----")
```

```python
{'Researcher': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_hxMmPTrjwqFmg6u6lB81c383', 'function': {'arguments': '{"query": "global AI software market size past 5 years"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 42, 'prompt_tokens': 218, 'total_tokens': 260}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5', 'finish_reason': 'tool_calls', 'logprobs': None}, name='Researcher', id='run-350ced4f-9b70-46c1-8fc1-5ab338e520fd-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'global AI software market size past 5 years'}, 'id': 'call_hxMmPTrjwqFmg6u6lB81c383', 'type': 'tool_call'}], usage_metadata={'input_tokens': 218, 'output_tokens': 42, 'total_tokens': 260})], 'sender': 'Researcher'}}
----
{'call_tool': {'messages': [ToolMessage(content='[{"url": "https://www.gartner.com/en/documents/5314863", "content": "Summary. AI software spending will grow to $297.9 billion by 2027. Over the next five years, market growth will accelerate from 17.8% to 20.4% in 2027, with a 19.1% CAGR. Government has the largest spend of over $70 million by 2027, but oil and gas is growing fastest with a 25.2% CAGR."}, {"url": "https://www.abiresearch.com/news-resources/chart-data/report-artificial-intelligence-market-size-global/", "content": "The Artificial Intelligence (AI) software market size will be valued at US$98 billion in 2024. Growing at a Compound Annual Growth Rate (CAGR) of 30%, the AI software market size will reach US$391.43 billion in 2030. Generative AI will be the fastest growing AI framework with a 49.7% CAGR over the market forecast period with foundation models ..."}, {"url": "https://www.statista.com/statistics/607716/worldwide-artificial-intelligence-market-revenues/", "content": "Get in touch with us now. , Jul 27, 2022. The global artificial intelligence (AI) software market is forecast to grow rapidly in the coming years, reaching around 126 billion U.S. dollars by 2025 ..."}, {"url": "https://explodingtopics.com/blog/ai-statistics", "content": "Industry verticals utilizing AI technology include tech-related sales, insurance, banking, telecom, healthcare, manufacturing, retail, and marketing to name a few.\\n$1.5 trillion\\n$489 billion\\n$554 billion\\n$215 billion\\n$453 billion\\n$87 billion\\n$2.76 trillion\\n$520 billion\\n$1.06 trillion\\n$109 billion\\n$3.42 trillion\\n$1.15 trillion\\n$2.26 trillion\\n$461 billion\\n$3.72 trillion\\n$951 billion\\n$8.4 trillion\\n$3.78 trillion\\n$535 billion\\n$95 billion\\n$7.47 trillion\\n$1.85 trillion\\n$3.99 trillion\\n$939 billion\\n$1.08 trillion\\n$216 billion\\n$2.13 trillion\\n$744 billion\\n$962 billion\\n$304 billion\\n$6.18 trillion\\n$2.23 trillion\\nThe self-driving car market is proving to be more than a novelty niche by grossing $173 billion in global revenue.\\n A computer\'s speech recognition capability and a smartphone\'s ability to geotag video content before sharing are each examples of ML.\\n2020\\n2021\\nChange\\n37%\\n50%\\n34%\\n57%\\n29%\\n31%\\n28%\\n48%\\n27%\\n27%\\n27%\\n46%\\n26%\\n22%\\n26%\\n34%\\n20%\\n40%\\n19%\\n44%\\n14%\\n31%\\n15%\\n1%\\n Here\'s a look at global revenue from the NLP market from 2017 to 2025:\\nRevenue\\n$3.18 billion\\n$5.07 billion\\n$8.21 billion\\n$12.4 billion\\n$17.58 billion\\n$24 billion\\n$30.35 billion\\n$37.33 billion\\n$43.29 billion\\n*Projected figures\\nDuring the forecast period of 2017-2025, the NLP market is set to grow by around 14x.\\n 93%\\n89%\\n85%\\n92%\\n68%\\n71%\\n98%\\n86%\\n72%\\n96%\\n76%\\n93%\\n86%\\n93%\\n69%\\n52% of telecommunications organizations utilize chatbots to increase their overall productivity.\\n The AI software market\'s global annual revenue (2018 to 2025):\\nRevenue\\n$10.1 billion\\n$14.69 billion\\n$22.59 billion\\n$34.87 billion\\n$51.27 billion\\n$70.94 billion\\n$94.41 billion\\n$126 billion\\n*Projected figures\\n"}, {"url": "https://www.gartner.com/en/newsroom/press-releases/2021-11-22-gartner-forecasts-worldwide-artificial-intelligence-software-market-to-reach-62-billion-in-2022", "content": "Market Growth Will Accelerate as Organizations Progress Their AI Maturity. Worldwide artificial intelligence (AI) software revenue is forecast to total $62.5 billion in 2022, an increase of 21.3% from 2021, according to a new forecast from Gartner, Inc. \\"The AI software market is picking up speed, but its long-term trajectory will depend on ..."}]', name='tavily_search_results_json', tool_call_id='call_hxMmPTrjwqFmg6u6lB81c383', artifact={'query': 'global AI software market size past 5 years', 'follow_up_questions': None, 'answer': None, 'images': [], 'results': [{'title': 'Forecast Analysis: AI Software Market by Vertical Industry, 2023-2027', 'url': 'https://www.gartner.com/en/documents/5314863', 'content': 'Summary. AI software spending will grow to $297.9 billion by 2027. Over the next five years, market growth will accelerate from 17.8% to 20.4% in 2027, with a 19.1% CAGR. Government has the largest spend of over $70 million by 2027, but oil and gas is growing fastest with a 25.2% CAGR.', 'score': 0.99712163, 'raw_content': None}, {'title': 'Artificial Intelligence (AI) Software Market Size: 2023 to 2030', 'url': 'https://www.abiresearch.com/news-resources/chart-data/report-artificial-intelligence-market-size-global/', 'content': 'The Artificial Intelligence (AI) software market size will be valued at US$98 billion in 2024. Growing at a Compound Annual Growth Rate (CAGR) of 30%, the AI software market size will reach US$391.43 billion in 2030. Generative AI will be the fastest growing AI framework with a 49.7% CAGR over the market forecast period with foundation models ...', 'score': 0.97754294, 'raw_content': None}, {'title': 'Global AI software market size 2018-2025 | Statista', 'url': 'https://www.statista.com/statistics/607716/worldwide-artificial-intelligence-market-revenues/', 'content': 'Get in touch with us now. , Jul 27, 2022. The global artificial intelligence (AI) software market is forecast to grow rapidly in the coming years, reaching around 126 billion U.S. dollars by 2025 ...', 'score': 0.9666632, 'raw_content': None}, {'title': '57 NEW Artificial Intelligence Statistics (Aug 2024) - Exploding Topics', 'url': 'https://explodingtopics.com/blog/ai-statistics', 'content': "Industry verticals utilizing AI technology include tech-related sales, insurance, banking, telecom, healthcare, manufacturing, retail, and marketing to name a few.\n$1.5 trillion\n$489 billion\n$554 billion\n$215 billion\n$453 billion\n$87 billion\n$2.76 trillion\n$520 billion\n$1.06 trillion\n$109 billion\n$3.42 trillion\n$1.15 trillion\n$2.26 trillion\n$461 billion\n$3.72 trillion\n$951 billion\n$8.4 trillion\n$3.78 trillion\n$535 billion\n$95 billion\n$7.47 trillion\n$1.85 trillion\n$3.99 trillion\n$939 billion\n$1.08 trillion\n$216 billion\n$2.13 trillion\n$744 billion\n$962 billion\n$304 billion\n$6.18 trillion\n$2.23 trillion\nThe self-driving car market is proving to be more than a novelty niche by grossing $173 billion in global revenue.\n A computer's speech recognition capability and a smartphone's ability to geotag video content before sharing are each examples of ML.\n2020\n2021\nChange\n37%\n50%\n34%\n57%\n29%\n31%\n28%\n48%\n27%\n27%\n27%\n46%\n26%\n22%\n26%\n34%\n20%\n40%\n19%\n44%\n14%\n31%\n15%\n1%\n Here's a look at global revenue from the NLP market from 2017 to 2025:\nRevenue\n$3.18 billion\n$5.07 billion\n$8.21 billion\n$12.4 billion\n$17.58 billion\n$24 billion\n$30.35 billion\n$37.33 billion\n$43.29 billion\n*Projected figures\nDuring the forecast period of 2017-2025, the NLP market is set to grow by around 14x.\n 93%\n89%\n85%\n92%\n68%\n71%\n98%\n86%\n72%\n96%\n76%\n93%\n86%\n93%\n69%\n52% of telecommunications organizations utilize chatbots to increase their overall productivity.\n The AI software market's global annual revenue (2018 to 2025):\nRevenue\n$10.1 billion\n$14.69 billion\n$22.59 billion\n$34.87 billion\n$51.27 billion\n$70.94 billion\n$94.41 billion\n$126 billion\n*Projected figures\n", 'score': 0.95022655, 'raw_content': None}, {'title': 'Gartner Forecasts Worldwide Artificial Intelligence Software Market to ...', 'url': 'https://www.gartner.com/en/newsroom/press-releases/2021-11-22-gartner-forecasts-worldwide-artificial-intelligence-software-market-to-reach-62-billion-in-2022', 'content': 'Market Growth Will Accelerate as Organizations Progress Their AI Maturity. Worldwide artificial intelligence (AI) software revenue is forecast to total $62.5 billion in 2022, an increase of 21.3% from 2021, according to a new forecast from Gartner, Inc. "The AI software market is picking up speed, but its long-term trajectory will depend on ...', 'score': 0.91535604, 'raw_content': None}], 'response_time': 1.96})]}}
----
{'Researcher': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_dmHbjJBuN6z6H1YhqVa1qT3X', 'function': {'arguments': '{\n  "type": "line",\n  "data": {\n    "labels": ["2018", "2019", "2020", "2021", "2022"],\n    "datasets": [{\n      "label": "AI Software Market Size (in billion USD)",\n      "data": [10.1, 14.69, 22.59, 34.87, 62.5],\n      "borderColor": "blue",\n      "fill": false\n    }]\n  },\n  "options": {\n    "title": {\n      "display": true,\n      "text": "Global AI Software Market Size (2018-2022)"\n    },\n    "scales": {\n      "yAxes": [{\n        "ticks": {\n          "beginAtZero": true\n        },\n        "scaleLabel": {\n          "display": true,\n          "labelString": "Market Size (in billion USD)"\n        }\n      }],\n      "xAxes": [{\n        "scaleLabel": {\n          "display": true,\n          "labelString": "Year"\n        }\n      }]\n    }\n  }\n}', 'name': 'chart_generator'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 235, 'prompt_tokens': 1269, 'total_tokens': 1504}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5', 'finish_reason': 'tool_calls', 'logprobs': None}, name='Researcher', id='run-76d4ceaa-eb77-44c8-a3c2-11c1bb795eab-0', tool_calls=[{'name': 'chart_generator', 'args': {'type': 'line', 'data': {'labels': ['2018', '2019', '2020', '2021', '2022'], 'datasets': [{'label': 'AI Software Market Size (in billion USD)', 'data': [10.1, 14.69, 22.59, 34.87, 62.5], 'borderColor': 'blue', 'fill': False}]}, 'options': {'title': {'display': True, 'text': 'Global AI Software Market Size (2018-2022)'}, 'scales': {'yAxes': [{'ticks': {'beginAtZero': True}, 'scaleLabel': {'display': True, 'labelString': 'Market Size (in billion USD)'}}], 'xAxes': [{'scaleLabel': {'display': True, 'labelString': 'Year'}}]}}}, 'id': 'call_dmHbjJBuN6z6H1YhqVa1qT3X', 'type': 'tool_call'}], usage_metadata={'input_tokens': 1269, 'output_tokens': 235, 'total_tokens': 1504})], 'sender': 'Researcher'}}
----
{'call_tool': {'messages': [ToolMessage(content='Error: chart_generator is not a valid tool, try one of [tavily_search_results_json, python_repl].', name='chart_generator', tool_call_id='call_dmHbjJBuN6z6H1YhqVa1qT3X')]}}
----
{'Researcher': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_r78t4mMNcNzQRFWrTBfGu2uc', 'function': {'arguments': '{"query":"global AI software market size past 5 years"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 26, 'prompt_tokens': 1464, 'total_tokens': 1490}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5', 'finish_reason': 'tool_calls', 'logprobs': None}, name='Researcher', id='run-9b4fb9c7-c8a5-4b91-86b5-de6d83200415-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'global AI software market size past 5 years'}, 'id': 'call_r78t4mMNcNzQRFWrTBfGu2uc', 'type': 'tool_call'}], usage_metadata={'input_tokens': 1464, 'output_tokens': 26, 'total_tokens': 1490})], 'sender': 'Researcher'}}
----
{'call_tool': {'messages': [ToolMessage(content='[{"url": "https://www.gartner.com/en/documents/5314863", "content": "Summary. AI software spending will grow to $297.9 billion by 2027. Over the next five years, market growth will accelerate from 17.8% to 20.4% in 2027, with a 19.1% CAGR. Government has the largest spend of over $70 million by 2027, but oil and gas is growing fastest with a 25.2% CAGR."}, {"url": "https://www.precedenceresearch.com/artificial-intelligence-software-market", "content": "Report Code : 2484. Category : ICT. The global artificial intelligence (AI) software market size was evaluated at USD 138.4 billion in 2022 and is predicted to hit around USD 1,094.52 billion by 2032, growing at a CAGR of 22.97% during the forecast period from 2023 to 2032."}, {"url": "https://www.statista.com/statistics/607716/worldwide-artificial-intelligence-market-revenues/", "content": "Get in touch with us now. , Jul 27, 2022. The global artificial intelligence (AI) software market is forecast to grow rapidly in the coming years, reaching around 126 billion U.S. dollars by 2025 ..."}, {"url": "https://explodingtopics.com/blog/ai-statistics", "content": "Industry verticals utilizing AI technology include tech-related sales, insurance, banking, telecom, healthcare, manufacturing, retail, and marketing to name a few.\\n$1.5 trillion\\n$489 billion\\n$554 billion\\n$215 billion\\n$453 billion\\n$87 billion\\n$2.76 trillion\\n$520 billion\\n$1.06 trillion\\n$109 billion\\n$3.42 trillion\\n$1.15 trillion\\n$2.26 trillion\\n$461 billion\\n$3.72 trillion\\n$951 billion\\n$8.4 trillion\\n$3.78 trillion\\n$535 billion\\n$95 billion\\n$7.47 trillion\\n$1.85 trillion\\n$3.99 trillion\\n$939 billion\\n$1.08 trillion\\n$216 billion\\n$2.13 trillion\\n$744 billion\\n$962 billion\\n$304 billion\\n$6.18 trillion\\n$2.23 trillion\\nThe self-driving car market is proving to be more than a novelty niche by grossing $173 billion in global revenue.\\n A computer\'s speech recognition capability and a smartphone\'s ability to geotag video content before sharing are each examples of ML.\\n2020\\n2021\\nChange\\n37%\\n50%\\n34%\\n57%\\n29%\\n31%\\n28%\\n48%\\n27%\\n27%\\n27%\\n46%\\n26%\\n22%\\n26%\\n34%\\n20%\\n40%\\n19%\\n44%\\n14%\\n31%\\n15%\\n1%\\n Here\'s a look at global revenue from the NLP market from 2017 to 2025:\\nRevenue\\n$3.18 billion\\n$5.07 billion\\n$8.21 billion\\n$12.4 billion\\n$17.58 billion\\n$24 billion\\n$30.35 billion\\n$37.33 billion\\n$43.29 billion\\n*Projected figures\\nDuring the forecast period of 2017-2025, the NLP market is set to grow by around 14x.\\n 93%\\n89%\\n85%\\n92%\\n68%\\n71%\\n98%\\n86%\\n72%\\n96%\\n76%\\n93%\\n86%\\n93%\\n69%\\n52% of telecommunications organizations utilize chatbots to increase their overall productivity.\\n The AI software market\'s global annual revenue (2018 to 2025):\\nRevenue\\n$10.1 billion\\n$14.69 billion\\n$22.59 billion\\n$34.87 billion\\n$51.27 billion\\n$70.94 billion\\n$94.41 billion\\n$126 billion\\n*Projected figures\\n"}, {"url": "https://www.gartner.com/en/newsroom/press-releases/2021-11-22-gartner-forecasts-worldwide-artificial-intelligence-software-market-to-reach-62-billion-in-2022", "content": "Market Growth Will Accelerate as Organizations Progress Their AI Maturity. Worldwide artificial intelligence (AI) software revenue is forecast to total $62.5 billion in 2022, an increase of 21.3% from 2021, according to a new forecast from Gartner, Inc. \\"The AI software market is picking up speed, but its long-term trajectory will depend on ..."}]', name='tavily_search_results_json', tool_call_id='call_r78t4mMNcNzQRFWrTBfGu2uc', artifact={'query': 'global AI software market size past 5 years', 'follow_up_questions': None, 'answer': None, 'images': [], 'results': [{'title': 'Forecast Analysis: AI Software Market by Vertical Industry, 2023-2027', 'url': 'https://www.gartner.com/en/documents/5314863', 'content': 'Summary. AI software spending will grow to $297.9 billion by 2027. Over the next five years, market growth will accelerate from 17.8% to 20.4% in 2027, with a 19.1% CAGR. Government has the largest spend of over $70 million by 2027, but oil and gas is growing fastest with a 25.2% CAGR.', 'score': 0.99712163, 'raw_content': None}, {'title': 'Artificial Intelligence (AI) Software Market Size, Report 2032', 'url': 'https://www.precedenceresearch.com/artificial-intelligence-software-market', 'content': 'Report Code : 2484. Category : ICT. The global artificial intelligence (AI) software market size was evaluated at USD 138.4 billion in 2022 and is predicted to hit around USD 1,094.52 billion by 2032, growing at a CAGR of 22.97% during the forecast period from 2023 to 2032.', 'score': 0.9797065, 'raw_content': None}, {'title': 'Global AI software market size 2018-2025 | Statista', 'url': 'https://www.statista.com/statistics/607716/worldwide-artificial-intelligence-market-revenues/', 'content': 'Get in touch with us now. , Jul 27, 2022. The global artificial intelligence (AI) software market is forecast to grow rapidly in the coming years, reaching around 126 billion U.S. dollars by 2025 ...', 'score': 0.9666632, 'raw_content': None}, {'title': '57 NEW Artificial Intelligence Statistics (Aug 2024) - Exploding Topics', 'url': 'https://explodingtopics.com/blog/ai-statistics', 'content': "Industry verticals utilizing AI technology include tech-related sales, insurance, banking, telecom, healthcare, manufacturing, retail, and marketing to name a few.\n$1.5 trillion\n$489 billion\n$554 billion\n$215 billion\n$453 billion\n$87 billion\n$2.76 trillion\n$520 billion\n$1.06 trillion\n$109 billion\n$3.42 trillion\n$1.15 trillion\n$2.26 trillion\n$461 billion\n$3.72 trillion\n$951 billion\n$8.4 trillion\n$3.78 trillion\n$535 billion\n$95 billion\n$7.47 trillion\n$1.85 trillion\n$3.99 trillion\n$939 billion\n$1.08 trillion\n$216 billion\n$2.13 trillion\n$744 billion\n$962 billion\n$304 billion\n$6.18 trillion\n$2.23 trillion\nThe self-driving car market is proving to be more than a novelty niche by grossing $173 billion in global revenue.\n A computer's speech recognition capability and a smartphone's ability to geotag video content before sharing are each examples of ML.\n2020\n2021\nChange\n37%\n50%\n34%\n57%\n29%\n31%\n28%\n48%\n27%\n27%\n27%\n46%\n26%\n22%\n26%\n34%\n20%\n40%\n19%\n44%\n14%\n31%\n15%\n1%\n Here's a look at global revenue from the NLP market from 2017 to 2025:\nRevenue\n$3.18 billion\n$5.07 billion\n$8.21 billion\n$12.4 billion\n$17.58 billion\n$24 billion\n$30.35 billion\n$37.33 billion\n$43.29 billion\n*Projected figures\nDuring the forecast period of 2017-2025, the NLP market is set to grow by around 14x.\n 93%\n89%\n85%\n92%\n68%\n71%\n98%\n86%\n72%\n96%\n76%\n93%\n86%\n93%\n69%\n52% of telecommunications organizations utilize chatbots to increase their overall productivity.\n The AI software market's global annual revenue (2018 to 2025):\nRevenue\n$10.1 billion\n$14.69 billion\n$22.59 billion\n$34.87 billion\n$51.27 billion\n$70.94 billion\n$94.41 billion\n$126 billion\n*Projected figures\n", 'score': 0.95022655, 'raw_content': None}, {'title': 'Gartner Forecasts Worldwide Artificial Intelligence Software Market to ...', 'url': 'https://www.gartner.com/en/newsroom/press-releases/2021-11-22-gartner-forecasts-worldwide-artificial-intelligence-software-market-to-reach-62-billion-in-2022', 'content': 'Market Growth Will Accelerate as Organizations Progress Their AI Maturity. Worldwide artificial intelligence (AI) software revenue is forecast to total $62.5 billion in 2022, an increase of 21.3% from 2021, according to a new forecast from Gartner, Inc. "The AI software market is picking up speed, but its long-term trajectory will depend on ...', 'score': 0.91535604, 'raw_content': None}], 'response_time': 2.11})]}}
----
{'Researcher': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_t8ZL0IeFIlqLOogieZF9rQLs', 'function': {'arguments': '{"query":"global AI software market size 2018, 2019, 2020, 2021, 2022"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 41, 'prompt_tokens': 2504, 'total_tokens': 2545}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5', 'finish_reason': 'tool_calls', 'logprobs': None}, name='Researcher', id='run-60b2faef-a310-4870-9cc4-fbb4b62c04a0-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'global AI software market size 2018, 2019, 2020, 2021, 2022'}, 'id': 'call_t8ZL0IeFIlqLOogieZF9rQLs', 'type': 'tool_call'}], usage_metadata={'input_tokens': 2504, 'output_tokens': 41, 'total_tokens': 2545})], 'sender': 'Researcher'}}
----
{'call_tool': {'messages': [ToolMessage(content='[{"url": "https://www.statista.com/statistics/607716/worldwide-artificial-intelligence-market-revenues/", "content": "Get in touch with us now. , Jul 27, 2022. The global artificial intelligence (AI) software market is forecast to grow rapidly in the coming years, reaching around 126 billion U.S. dollars by 2025 ..."}, {"url": "https://www.gartner.com/en/newsroom/press-releases/2021-11-22-gartner-forecasts-worldwide-artificial-intelligence-software-market-to-reach-62-billion-in-2022", "content": "Market Growth Will Accelerate as Organizations Progress Their AI Maturity Worldwide artificial intelligence (AI) software revenue is forecast to total $62.5 billion in 2022, an increase of 21.3% from 2021, according to a new forecast from Gartner, Inc."}, {"url": "https://www.gartner.com/en/documents/4007140", "content": "Summary By 2025, the market for artificial intelligence (AI) software will reach almost $134.8 billion. Over the next five years, the market growth will accelerate from 14.4% in 2021 to reach 31.1% in 2025, considerably outpacing the overall software market growth."}, {"url": "https://www.grandviewresearch.com/industry-analysis/artificial-intelligence-ai-market", "content": "Artificial Intelligence Market Report Scope\\nReport Attribute\\nDetails\\nMarket size value in 2023\\nUSD 196.63 billion\\nRevenue forecast in 2030\\nUSD 1,811.75 billion\\nGrowth rate\\nCAGR of 37.3% from 2023 to 2030\\nBase year for estimation\\n2022\\nHistorical data\\n2017 - 2021\\nForecast period\\n2023 - 2030\\nQuantitative units\\nRevenue in USD billion, CAGR from 2023 to 2030\\nReport coverage\\nRevenue forecast, company ranking, competitive landscape, growth factors, and trends\\nSegments covered\\nSolution, technology, end-use, region\\nRegional scope\\nNorth America; Europe; Asia Pacific; South America; MEA\\nCountry scope\\nU.S.; Canada; Mexico; Germany; U.K.; China; Japan; India; Brazil\\nKey companies profiled\\nAdvanced Micro Devices; AiCure; Arm Limited; Atomwise, Inc.; Ayasdi AI LLC; Baidu, Inc.; Clarifai, Inc; Cyrcadia Health; Enlitic, Inc.; Google LLC; H2O.ai.; HyperVerge, Inc.; International Business Machines Corporation; IBM Watson Health; Intel Corporation; Iris.ai AS.; Deep Learning\\nMachine Learning\\nNatural Language Processing (NLP)\\nMachine Vision\\nEnd-use Outlook (Revenue, USD Billion, 2017 - 2030)\\nHealthcare\\nRobot-Assisted Surgery\\nVirtual Nursing Assistants\\nHospital Workflow Management\\nDosage Error Reduction\\nClinical Trial Participant Identifier\\nPreliminary Diagnosis\\nAutomated Image Diagnosis\\nBFSI\\nRisk Assessment\\nFinancial Analysis/Research\\nInvestment/Portfolio Management\\nOthers\\nLaw\\nRetail\\nAdvertising & Media\\nAutomotive & Transportation\\nAgriculture\\nManufacturing\\nOthers\\nRegional Outlook (Revenue, USD Billion, 2017 - 2030)\\n North America\\nU.S.\\nCanada\\nMexico\\nEurope\\nGermany\\nU.K.\\nAsia Pacific\\nChina\\nJapan\\nIndia\\nSouth America\\nBrazil\\nMiddle East and Africa (MEA)\\nFrequently Asked Questions About This Report\\nb.\\n b.\\nKey factors that are driving the artificial intelligence market growth include a rise in the adoption of big data, analytics, and the increasing potential of R&D in developing AI systems and technological innovations across the globe.\\n Artificial Intelligence Market Size, Share & Trends Analysis Report By Solution, By Technology (Deep Learning, Machine Learning), By End-use, By Region, And Segment Forecasts, 2023 - 2030\\nReport Overview\\nThe global artificial intelligence market size was valued at USD 136.55 billion in 2022 and is projected to expand at a compound annual growth rate (CAGR) of 37.3% from 2023 to 2030."}, {"url": "https://www.marketsandmarkets.com/Market-Reports/artificial-intelligence-market-74851580.html", "content": "Get online access to the report on the World\'s First Market Intelligence Cloud\\nEasy to Download Historical Data & Forecast Numbers\\nCompany Analysis Dashboard for high growth potential opportunities\\nResearch Analyst Access for customization & queries\\nCompetitor Analysis with Interactive dashboard\\nLatest News, Updates & Trend analysisRequest Sample\\nClick on image to enlarge\\nGet online access to the report on the World\'s First Market Intelligence Cloud\\nScope of the Report\\nReport Metrics\\nDetails\\nMarket size value in 2023\\nUS $150.2 billion\\nMarket Revenue in 2030\\nUS $1,345.2 billion\\nGrowth Rate\\nCAGR of 36.8% from 2023 to 2030\\nArtificial Intelligence Market Drivers\\nMarket size available for years\\n2017\\u20132030\\nBase year considered\\n2022\\nForecast period\\n2023\\u20132030\\nForecast units\\nUSD (Billion)\\nSegments covered\\nOffering, Technology, Business Function, Verticals, and Region\\nGeographies covered\\nNorth America, Europe, Asia Pacific, Middle East & Africa, and Latin America\\nCompanies covered\\nGoogle (US), Microsoft (US), IBM (US), Oracle (US), AWS (US), Intel (US), Salesforce (US), SAP (Germany), Cisco (US), Meta (US), HPE (US), Siemens (Germany), Huawei (China), NVIDIA (US), Baidu (China), SAS Institute (US), OpenAI (US), H2O.ai (US), iFLYTEK (China), Alibaba Cloud (China), General Vision (US), Darktrace (UK), Blackberry Limited (Canada), DiDi Global (China), Face++ (China), Inbenta (US), Anju Software (US), Butterfly Network (US), Atomwise (US), AIBrain (US), SK Hynix (South Korea), Progress (US), PrecisionHawk (US), AgEagle Aerial System (US), Neurala (US), Twitter (US), Aurea Software (US), Persado (US), 8x8 (US), Appier (Taiwan), GumGum (US), IPRO (US), Graphcore (UK), Preferred Networks (Japan), Applied Brain Research (Canada), Pilot AI (US), Iris Automation (US), Gamaya (Switzerland), ec2ce (Spain), Descartes Labs (US), Mythic (US), Ada (Canada), Mostly AI (Austria), Sentient.io (Singapore), Lumen5 (Canada), AI Superior (Germany), Fosfor (India), Intrinsic (US), Jasper (US), Soundful (US), Writesonic (US), One AI (Israel).\\n The major players in the artificial intelligence market include Google (US), Microsoft (US), IBM (US), Oracle (US), AWS (US), Intel (US), Salesforce (US), SAP (Germany), Cisco (US), Meta (US), HPE (US), Siemens (Germany), Huawei (China), NVIDIA (US), Baidu (China), SAS Institute (US), OpenAI (US), H2O.ai (US), iFLYTEK (China), Alibaba Cloud (China), General Vision (US), Darktrace (UK), Blackberry Limited (Canada), DiDi Global (China), Face++ (China), Inbenta (US), Anju Software (US), Butterfly Network (US), Atomwise (US), AIBrain (US), SK Hynix (South Korea), Progress (US), PrecisionHawk (US), AgEagle Aerial System (US), Neurala (US), Twitter (US), Aurea Software (US), Persado (US), 8x8 (US), Appier (Taiwan), GumGum (US), IPRO (US), Graphcore (UK), Preferred Networks (Japan), Applied Brain Research (Canada), Pilot AI (US), Iris Automation (US), Gamaya (Switzerland), ec2ce (Spain), Descartes Labs (US), Mythic (US), Ada (Canada), Mostly AI (Austria), Sentient.io (Singapore), Lumen5 (Canada), AI Superior (Germany), Fosfor (India), Intrinsic (US), Jasper (US), Soundful (US), Writesonic (US), One AI (Israel).\\n Information & Communications Technology\\nSemiconductor & Electronics\\nMedical Devices\\nChemicals\\nAdvanced Materials\\nEnergy and Power\\nFood and Beverage\\nBiotechnology\\nAerospace & Defence\\nAutomotive & Transportation\\nAutomation & Process Control\\nAgriculture Industry\\nPackaging\\nPharmaceuticals\\nHealthcare IT\\nPayer IT\\nPharmaceutical IT\\nProvider IT\\nAnalytical and Scientific Instrumentation\\nBuilding & Construction\\nMining, Minerals and Metals\\nEngineering Equipment and Devices The following is the breakup of primary profiles:\\nTo know about the assumptions considered for the study, download the pdf brochure\\nMarket Size Estimation\\nTo know about the assumptions considered for the study, Request for Free Sample Report\\nTop-down and bottom-up approaches were adopted for estimating and forecasting the artificial intelligence market. To know about the assumptions considered for the study, Request for Free Sample Report\\nTo know about the assumptions considered for the study, download the pdf brochure\\nArtificial Intelligence Market Growth Dynamics\\nThe increasing adoption of autonomous artificial intelligence (AI) is driving the growth of the Artificial Intelligence Market."}]', name='tavily_search_results_json', tool_call_id='call_t8ZL0IeFIlqLOogieZF9rQLs', artifact={'query': 'global AI software market size 2018, 2019, 2020, 2021, 2022', 'follow_up_questions': None, 'answer': None, 'images': [], 'results': [{'title': 'Global AI software market size 2018-2025', 'url': 'https://www.statista.com/statistics/607716/worldwide-artificial-intelligence-market-revenues/', 'content': 'Get in touch with us now. , Jul 27, 2022. The global artificial intelligence (AI) software market is forecast to grow rapidly in the coming years, reaching around 126 billion U.S. dollars by 2025 ...', 'score': 0.99901885, 'raw_content': None}, {'title': 'Gartner Forecasts Worldwide Artificial Intelligence Software Market to ...', 'url': 'https://www.gartner.com/en/newsroom/press-releases/2021-11-22-gartner-forecasts-worldwide-artificial-intelligence-software-market-to-reach-62-billion-in-2022', 'content': 'Market Growth Will Accelerate as Organizations Progress Their AI Maturity Worldwide artificial intelligence (AI) software revenue is forecast to total $62.5 billion in 2022, an increase of 21.3% from 2021, according to a new forecast from Gartner, Inc.', 'score': 0.99896777, 'raw_content': None}, {'title': 'Forecast Analysis: Artificial Intelligence Software, Worldwide', 'url': 'https://www.gartner.com/en/documents/4007140', 'content': 'Summary By 2025, the market for artificial intelligence (AI) software will reach almost $134.8 billion. Over the next five years, the market growth will accelerate from 14.4% in 2021 to reach 31.1% in 2025, considerably outpacing the overall software market growth.', 'score': 0.9984146, 'raw_content': None}, {'title': 'Artificial Intelligence Market Size, Share, Growth Report 2030', 'url': 'https://www.grandviewresearch.com/industry-analysis/artificial-intelligence-ai-market', 'content': 'Artificial Intelligence Market Report Scope\nReport Attribute\nDetails\nMarket size value in 2023\nUSD 196.63 billion\nRevenue forecast in 2030\nUSD 1,811.75 billion\nGrowth rate\nCAGR of 37.3% from 2023 to 2030\nBase year for estimation\n2022\nHistorical data\n2017 - 2021\nForecast period\n2023 - 2030\nQuantitative units\nRevenue in USD billion, CAGR from 2023 to 2030\nReport coverage\nRevenue forecast, company ranking, competitive landscape, growth factors, and trends\nSegments covered\nSolution, technology, end-use, region\nRegional scope\nNorth America; Europe; Asia Pacific; South America; MEA\nCountry scope\nU.S.; Canada; Mexico; Germany; U.K.; China; Japan; India; Brazil\nKey companies profiled\nAdvanced Micro Devices; AiCure; Arm Limited; Atomwise, Inc.; Ayasdi AI LLC; Baidu, Inc.; Clarifai, Inc; Cyrcadia Health; Enlitic, Inc.; Google LLC; H2O.ai.; HyperVerge, Inc.; International Business Machines Corporation; IBM Watson Health; Intel Corporation; Iris.ai AS.; Deep Learning\nMachine Learning\nNatural Language Processing (NLP)\nMachine Vision\nEnd-use Outlook (Revenue, USD Billion, 2017 - 2030)\nHealthcare\nRobot-Assisted Surgery\nVirtual Nursing Assistants\nHospital Workflow Management\nDosage Error Reduction\nClinical Trial Participant Identifier\nPreliminary Diagnosis\nAutomated Image Diagnosis\nBFSI\nRisk Assessment\nFinancial Analysis/Research\nInvestment/Portfolio Management\nOthers\nLaw\nRetail\nAdvertising & Media\nAutomotive & Transportation\nAgriculture\nManufacturing\nOthers\nRegional Outlook (Revenue, USD Billion, 2017 - 2030)\n North America\nU.S.\nCanada\nMexico\nEurope\nGermany\nU.K.\nAsia Pacific\nChina\nJapan\nIndia\nSouth America\nBrazil\nMiddle East and Africa (MEA)\nFrequently Asked Questions About This Report\nb.\n b.\nKey factors that are driving the artificial intelligence market growth include a rise in the adoption of big data, analytics, and the increasing potential of R&D in developing AI systems and technological innovations across the globe.\n Artificial Intelligence Market Size, Share & Trends Analysis Report By Solution, By Technology (Deep Learning, Machine Learning), By End-use, By Region, And Segment Forecasts, 2023 - 2030\nReport Overview\nThe global artificial intelligence market size was valued at USD 136.55 billion in 2022 and is projected to expand at a compound annual growth rate (CAGR) of 37.3% from 2023 to 2030.', 'score': 0.9953544, 'raw_content': None}, {'title': 'Artificial Intelligence Market Size & Trends, Growth Analysis, Forecast ...', 'url': 'https://www.marketsandmarkets.com/Market-Reports/artificial-intelligence-market-74851580.html', 'content': "Get online access to the report on the World's First Market Intelligence Cloud\nEasy to Download Historical Data & Forecast Numbers\nCompany Analysis Dashboard for high growth potential opportunities\nResearch Analyst Access for customization & queries\nCompetitor Analysis with Interactive dashboard\nLatest News, Updates & Trend analysisRequest Sample\nClick on image to enlarge\nGet online access to the report on the World's First Market Intelligence Cloud\nScope of the Report\nReport Metrics\nDetails\nMarket size value in 2023\nUS $150.2 billion\nMarket Revenue in 2030\nUS $1,345.2 billion\nGrowth Rate\nCAGR of 36.8% from 2023 to 2030\nArtificial Intelligence Market Drivers\nMarket size available for years\n2017–2030\nBase year considered\n2022\nForecast period\n2023–2030\nForecast units\nUSD (Billion)\nSegments covered\nOffering, Technology, Business Function, Verticals, and Region\nGeographies covered\nNorth America, Europe, Asia Pacific, Middle East & Africa, and Latin America\nCompanies covered\nGoogle (US), Microsoft (US), IBM (US), Oracle (US), AWS (US), Intel (US), Salesforce (US), SAP (Germany), Cisco (US), Meta (US), HPE (US), Siemens (Germany), Huawei (China), NVIDIA (US), Baidu (China), SAS Institute (US), OpenAI (US), H2O.ai (US), iFLYTEK (China), Alibaba Cloud (China), General Vision (US), Darktrace (UK), Blackberry Limited (Canada), DiDi Global (China), Face++ (China), Inbenta (US), Anju Software (US), Butterfly Network (US), Atomwise (US), AIBrain (US), SK Hynix (South Korea), Progress (US), PrecisionHawk (US), AgEagle Aerial System (US), Neurala (US), Twitter (US), Aurea Software (US), Persado (US), 8x8 (US), Appier (Taiwan), GumGum (US), IPRO (US), Graphcore (UK), Preferred Networks (Japan), Applied Brain Research (Canada), Pilot AI (US), Iris Automation (US), Gamaya (Switzerland), ec2ce (Spain), Descartes Labs (US), Mythic (US), Ada (Canada), Mostly AI (Austria), Sentient.io (Singapore), Lumen5 (Canada), AI Superior (Germany), Fosfor (India), Intrinsic (US), Jasper (US), Soundful (US), Writesonic (US), One AI (Israel).\n The major players in the artificial intelligence market include Google (US), Microsoft (US), IBM (US), Oracle (US), AWS (US), Intel (US), Salesforce (US), SAP (Germany), Cisco (US), Meta (US), HPE (US), Siemens (Germany), Huawei (China), NVIDIA (US), Baidu (China), SAS Institute (US), OpenAI (US), H2O.ai (US), iFLYTEK (China), Alibaba Cloud (China), General Vision (US), Darktrace (UK), Blackberry Limited (Canada), DiDi Global (China), Face++ (China), Inbenta (US), Anju Software (US), Butterfly Network (US), Atomwise (US), AIBrain (US), SK Hynix (South Korea), Progress (US), PrecisionHawk (US), AgEagle Aerial System (US), Neurala (US), Twitter (US), Aurea Software (US), Persado (US), 8x8 (US), Appier (Taiwan), GumGum (US), IPRO (US), Graphcore (UK), Preferred Networks (Japan), Applied Brain Research (Canada), Pilot AI (US), Iris Automation (US), Gamaya (Switzerland), ec2ce (Spain), Descartes Labs (US), Mythic (US), Ada (Canada), Mostly AI (Austria), Sentient.io (Singapore), Lumen5 (Canada), AI Superior (Germany), Fosfor (India), Intrinsic (US), Jasper (US), Soundful (US), Writesonic (US), One AI (Israel).\n Information & Communications Technology\nSemiconductor & Electronics\nMedical Devices\nChemicals\nAdvanced Materials\nEnergy and Power\nFood and Beverage\nBiotechnology\nAerospace & Defence\nAutomotive & Transportation\nAutomation & Process Control\nAgriculture Industry\nPackaging\nPharmaceuticals\nHealthcare IT\nPayer IT\nPharmaceutical IT\nProvider IT\nAnalytical and Scientific Instrumentation\nBuilding & Construction\nMining, Minerals and Metals\nEngineering Equipment and Devices The following is the breakup of primary profiles:\nTo know about the assumptions considered for the study, download the pdf brochure\nMarket Size Estimation\nTo know about the assumptions considered for the study, Request for Free Sample Report\nTop-down and bottom-up approaches were adopted for estimating and forecasting the artificial intelligence market. To know about the assumptions considered for the study, Request for Free Sample Report\nTo know about the assumptions considered for the study, download the pdf brochure\nArtificial Intelligence Market Growth Dynamics\nThe increasing adoption of autonomous artificial intelligence (AI) is driving the growth of the Artificial Intelligence Market.", 'score': 0.99087435, 'raw_content': None}], 'response_time': 2.13})]}}
----
{'Researcher': {'messages': [AIMessage(content='Here is the data for the global AI software market size over the past five years:\n\n- 2018: $10.1 billion\n- 2019: $14.69 billion\n- 2020: $22.59 billion\n- 2021: $34.87 billion\n- 2022: $62.5 billion\n\n### Python Code to Generate the Line Chart\n\n```python\nimport matplotlib.pyplot as plt\n\n# Data\nyears = ["2018", "2019", "2020", "2021", "2022"]\nmarket_size = [10.1, 14.69, 22.59, 34.87, 62.5]\n\n# Create the plot\nplt.figure(figsize=(10, 6))\nplt.plot(years, market_size, marker=\'o\', linestyle=\'-\', color=\'b\')\n\n# Adding titles and labels\nplt.title("Global AI Software Market Size (2018-2022)")\nplt.xlabel("Year")\nplt.ylabel("Market Size (in billion USD)")\nplt.grid(True)\n\n# Display the plot\nplt.show()\n```\n\nThis script will generate a line chart showing the growth of the global AI software market size from 2018 to 2022.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 253, 'prompt_tokens': 4562, 'total_tokens': 4815}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5', 'finish_reason': 'stop', 'logprobs': None}, name='Researcher', id='run-9c0880bd-f4fa-40a6-aa5b-0effe7d528e2-0', usage_metadata={'input_tokens': 4562, 'output_tokens': 253, 'total_tokens': 4815})], 'sender': 'Researcher'}}
----
{'chart_generator': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_FuausN6lrQVlOhEPUHyfvJQf', 'function': {'arguments': '{"code":"import matplotlib.pyplot as plt\\n\\n# Data\\nyears = [\\"2018\\", \\"2019\\", \\"2020\\", \\"2021\\", \\"2022\\"]\\nmarket_size = [10.1, 14.69, 22.59, 34.87, 62.5]\\n\\n# Create the plot\\nplt.figure(figsize=(10, 6))\\nplt.plot(years, market_size, marker=\'o\', linestyle=\'-\', color=\'b\')\\n\\n# Adding titles and labels\\nplt.title(\\"Global AI Software Market Size (2018-2022)\\")\\nplt.xlabel(\\"Year\\")\\nplt.ylabel(\\"Market Size (in billion USD)\\")\\nplt.grid(True)\\n\\n# Display the plot\\nplt.show()"}', 'name': 'python_repl'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 176, 'prompt_tokens': 4823, 'total_tokens': 4999}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_a2ff031fb5', 'finish_reason': 'tool_calls', 'logprobs': None}, name='chart_generator', id='run-90f8c5aa-4c1f-41fc-9ae4-dfadd0304c73-0', tool_calls=[{'name': 'python_repl', 'args': {'code': 'import matplotlib.pyplot as plt\n\n# Data\nyears = ["2018", "2019", "2020", "2021", "2022"]\nmarket_size = [10.1, 14.69, 22.59, 34.87, 62.5]\n\n# Create the plot\nplt.figure(figsize=(10, 6))\nplt.plot(years, market_size, marker=\'o\', linestyle=\'-\', color=\'b\')\n\n# Adding titles and labels\nplt.title("Global AI Software Market Size (2018-2022)")\nplt.xlabel("Year")\nplt.ylabel("Market Size (in billion USD)")\nplt.grid(True)\n\n# Display the plot\nplt.show()'}, 'id': 'call_FuausN6lrQVlOhEPUHyfvJQf', 'type': 'tool_call'}], usage_metadata={'input_tokens': 4823, 'output_tokens': 176, 'total_tokens': 4999})], 'sender': 'chart_generator'}}
----
Python REPL can execute arbitrary code. Use with caution.
{'call_tool': {'messages': [ToolMessage(content='成功执行:\n```python\nimport matplotlib.pyplot as plt\n\n# Data\nyears = ["2018", "2019", "2020", "2021", "2022"]\nmarket_size = [10.1, 14.69, 22.59, 34.87, 62.5]\n\n# Create the plot\nplt.figure(figsize=(10, 6))\nplt.plot(years, market_size, marker=\'o\', linestyle=\'-\', color=\'b\')\n\n# Adding titles and labels\nplt.title("Global AI Software Market Size (2018-2022)")\nplt.xlabel("Year")\nplt.ylabel("Market Size (in billion USD)")\nplt.grid(True)\n\n# Display the plot\nplt.show()\n```\nStdout: \n\n如果你已完成所有任务，请回复FINAL ANSWER。', name='python_repl', tool_call_id='call_FuausN6lrQVlOhEPUHyfvJQf')]}}
----
{'chart_generator': {'messages': [AIMessage(content='FINAL ANSWER\n\n以下是全球AI软件市场规模在2018年至2022年的变化折线图：\n\n![Global AI Software Market Size (2018-2022)](attachment://image.png)\n\n图表展示了过去五年全球AI软件市场规模的显著增长。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 60, 'prompt_tokens': 5175, 'total_tokens': 5235}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_a2ff031fb5', 'finish_reason': 'stop', 'logprobs': None}, name='chart_generator', id='run-c6fd2097-ad78-4a21-b47f-13035931d3af-0', usage_metadata={'input_tokens': 5175, 'output_tokens': 60, 'total_tokens': 5235})], 'sender': 'chart_generator'}}
----

```

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1724830525053-5f37eb2f-0755-4b9b-8e34-1e24efccafc4.png)



## Planning Agent
### Plan-and-Execute(计划并执行)
<font style="color:rgba(0, 0, 0, 0.87);">下面展示了如何创建一个“计划并执行”风格的代理。 这在很大程度上借鉴了 </font>[<font style="color:rgba(0, 0, 0, 0.87);">计划和解决</font>](https://arxiv.org/abs/2305.04091)<font style="color:rgba(0, 0, 0, 0.87);"> 论文以及 </font>[<font style="color:rgba(0, 0, 0, 0.87);">Baby-AGI</font>](https://github.com/yoheinakajima/babyagi)<font style="color:rgba(0, 0, 0, 0.87);"> 项目。</font>

<font style="color:rgba(0, 0, 0, 0.87);">核心思想是先制定一个多步骤计划，然后逐项执行。 完成一项特定任务后，您可以重新审视计划并根据需要进行修改。</font>

<font style="color:rgba(0, 0, 0, 0.87);">一般的计算图如下所示</font>

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1723625989932-714592f3-2767-4c21-babd-c1f40d570d54.png)<font style="color:rgba(0, 0, 0, 0.87);">这与典型的 </font>[<font style="color:rgba(0, 0, 0, 0.87);">ReAct</font>](https://arxiv.org/abs/2210.03629)<font style="color:rgba(0, 0, 0, 0.87);"> 风格的代理进行了比较，在该代理中，您一次思考一步。 这种“计划并执行”风格代理的优势在于</font>

1. <font style="color:rgba(0, 0, 0, 0.87);">明确的长期规划（即使是真正强大的 LLM 也可能难以做到）</font>
2. <font style="color:rgba(0, 0, 0, 0.87);">能够使用更小/更弱的模型来执行步骤，仅在规划步骤中使用更大/更好的模型</font>

<font style="color:rgba(0, 0, 0, 0.87);">以下演练演示了如何在 LangGraph 中实现这一点。 生成的代理将留下类似以下示例的轨迹： (</font>[<font style="color:rgba(0, 0, 0, 0.87);">链接</font>](https://smith.langchain.com/public/d46e24d3-dda6-44d5-9550-b618fca4e0d4/r)<font style="color:rgba(0, 0, 0, 0.87);">).</font>

#### <font style="color:rgba(0, 0, 0, 0.87);">设置</font>
<font style="color:rgba(0, 0, 0, 0.87);">首先，我们需要安装所需的软件包。</font>

```python

%pip install --quiet -U langgraph langchain-community langchain-openai tavily-python
```

<font style="color:rgba(0, 0, 0, 0.87);">接下来，我们需要为 OpenAI（我们将使用的 LLM）和 Tavily（我们将使用的搜索工具）设置 API 密钥</font>

<font style="color:rgba(0, 0, 0, 0.87);">可以选择设置 LangSmith 跟踪的 API 密钥，这将为我们提供一流的可观察性。</font>

```python
setx TAVILY_API_KEY ""
# Optional, add tracing in LangSmith
setx LANGCHAIN_TRACING_V2 "true"
setx LANGCHAIN_API_KEY ""
```

#### <font style="color:rgba(0, 0, 0, 0.87);">定义工具</font>
<font style="color:rgba(0, 0, 0, 0.87);">我们将首先定义要使用的工具。 对于这个简单的示例，我们将使用 Tavily 内置的搜索工具。 但是，创建自己的工具非常容易 - 请参阅有关如何操作的文档</font><font style="color:rgba(0, 0, 0, 0.87);"> </font>[<font style="color:rgba(0, 0, 0, 0.87);">此处</font>](http://python.langchain.ac.cn/v0.2/docs/how_to/custom_tools)<font style="color:rgba(0, 0, 0, 0.87);">。</font>

```python
#示例：plan_execute.py
from langchain_community.tools.tavily_search import TavilySearchResults
# 创建TavilySearchResults工具，设置最大结果数为1
tools = [TavilySearchResults(max_results=1)]
```

#### <font style="color:rgba(0, 0, 0, 0.87);">定义我们的执行代理</font>
<font style="color:rgba(0, 0, 0, 0.87);">现在我们将创建要用于执行任务的执行代理。 请注意，对于此示例，我们将对每个任务使用相同的执行代理，但这并非必须如此。</font>

```python
from langchain import hub
from langchain_openai import ChatOpenAI
import asyncio
from langgraph.prebuilt import create_react_agent

# 从LangChain的Hub中获取prompt模板，可以进行修改
prompt = hub.pull("wfh/react-agent-executor")
prompt.pretty_print()

# 选择驱动代理的LLM，使用OpenAI的ChatGPT-4o模型
llm = ChatOpenAI(model="gpt-4o")
# 创建一个REACT代理执行器，使用指定的LLM和工具，并应用从Hub中获取的prompt
agent_executor = create_react_agent(llm, tools, messages_modifier=prompt)
```

```python
================================ System Message ================================

You are a helpful assistant.

============================= Messages Placeholder =============================

{{messages}}
```

```python
# 调用代理执行器，询问“谁是美国公开赛的冠军”
agent_executor.invoke({"messages": [("user", "谁是美国公开赛的获胜者")]})
```

```python
{'messages': [HumanMessage(content='who is the winnner of the us open', id='7c491c9f-cdbe-4761-b93b-3e4eeb526c97'),
              AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_MMmwmxwxRH2hrmMbuBeMGsXW', 'function': {'arguments': '{"query":"US Open 2023 winner"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 23, 'prompt_tokens': 97, 'total_tokens': 120}, 'model_name': 'gpt-4-turbo-preview', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-855f7cff-62a2-4dd8-b71b-707b507b00a4-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'US Open 2023 winner'}, 'id': 'call_MMmwmxwxRH2hrmMbuBeMGsXW'}]),
              ToolMessage(content='[{"url": "https://www.bbc.com/sport/tennis/66766337", "content": ": Stephen Nolan goes in to find out\\nRelated Topics\\nTop Stories\\nTen Hag on Rashford plus transfer news, WSL deadline day\\nSpinner Leach doubtful for second Test in India\\nMcIlroy \'changes tune\' on LIV players\' punishment\\nElsewhere on the BBC\\nDiscover the tropical paradise of Thailand\\nFrom the secrets of the South to the mysterious North...\\n Djokovic offered to help up Medvedev when the Russian fell to the court in the third set\\nDjokovic\'s relentless returning continued to draw mistakes out of Medvedev, who was serving poorly and making loose errors, at the start of the second set.\\n It was clear to see Medvedev had needed to level by taking that second set to stand any real chance of victory and the feeling of the inevitable was heightened by the Russian needing treatment on a shoulder injury before the third set.\\n Djokovic shows again why he can never be written off\\nWhen Djokovic lost to 20-year-old Carlos Alcaraz in the Wimbledon final it felt like a changing-of-the-guard moment in the men\'s game.\\n The inside story of Putin\\u2019s invasion of Ukraine\\nTold by the Presidents and Prime Ministers tasked with making the critical decisions\\nSurvival of the wittiest!\\n"}, {"url": "https://www.usopen.org/en_US/news/articles/2023-09-10/novak_djokovic_wins_24th_grand_slam_singles_title_at_2023_us_open.html", "content": "WHAT HAPPENED: Novak Djokovic handled the weight of history to defeat Daniil Medvedev on Sunday in the 2023 US Open men\'s singles final. With a 6-3, 7-6(5), 6-3 victory, the 36-year-old won his 24th Grand Slam singles title, tying Margaret Court\'s record and bolstering his case to be considered the greatest tennis player of all time."}, {"url": "https://apnews.com/article/us-open-final-live-updates-djokovic-medvedev-8a4a26f8d77ef9ab2fb3efe1096dce7e", "content": "Novak Djokovic wins the US Open for his 24th Grand Slam title by beating Daniil Medvedev\\nNovak Djokovic, of Serbia, holds up the championship trophy after defeating Daniil Medvedev, of Russia, in the men\\u2019s singles final of the U.S. Open tennis championships, Sunday, Sept. 10, 2023, in New York. (AP Photo/Manu Fernandez)\\nDaniil Medvedev, of Russia, sits on the court after a rally against Novak Djokovic, of Serbia, during the men\\u2019s singles final of the U.S. Open tennis championships, Sunday, Sept. 10, 2023, in New York. (AP Photo/Manu Fernandez)\\nDaniil Medvedev, of Russia, sits on the court after a rally against Novak Djokovic, of Serbia, during the men\\u2019s singles final of the U.S. Open tennis championships, Sunday, Sept. 10, 2023, in New York. (AP Photo/Manu Fernandez)\\nDaniil Medvedev, of Russia, sits on the court after a rally against Novak Djokovic, of Serbia, during the men\\u2019s singles final of the U.S. Open tennis championships, Sunday, Sept. 10, 2023, in New York. Novak Djokovic, of Serbia, reveals a t-shirt honoring the number 24 and Kobe Bryant after defeating Daniil Medvedev, of Russia, in the men\\u2019s singles final of the U.S. Open tennis championships, Sunday, Sept. 10, 2023, in New York."}]', name='tavily_search_results_json', id='ca0ff812-6c7f-43c1-9d0e-427cfe8da332', tool_call_id='call_MMmwmxwxRH2hrmMbuBeMGsXW'),
  AIMessage(content="The winner of the 2023 US Open men's singles was Novak Djokovic. He defeated Daniil Medvedev with a score of 6-3, 7-6(5), 6-3 in the final, winning his 24th Grand Slam singles title. This victory tied Margaret Court's record and bolstered Djokovic's claim to be considered one of the greatest tennis players of all time.", response_metadata={'token_usage': {'completion_tokens': 89, 'prompt_tokens': 972, 'total_tokens': 1061}, 'model_name': 'gpt-4-turbo-preview', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-ef37a655-1ea6-470e-a310-8f125ca48015-0')]}
```

#### <font style="color:rgba(0, 0, 0, 0.87);">定义状态</font>
<font style="color:rgba(0, 0, 0, 0.87);">现在让我们从定义要跟踪此代理的状态开始。</font>

<font style="color:rgba(0, 0, 0, 0.87);">首先，我们需要跟踪当前计划。 让我们将其表示为字符串列表。</font>

<font style="color:rgba(0, 0, 0, 0.87);">接下来，我们应该跟踪先前执行的步骤。 让我们将其表示为元组列表（这些元组将包含步骤及其结果）</font>

<font style="color:rgba(0, 0, 0, 0.87);">最后，我们需要一些状态来表示最终响应以及原始输入。</font>

```python
import operator
from typing import Annotated, List, Tuple, TypedDict


# 定义一个TypedDict类PlanExecute，用于存储输入、计划、过去的步骤和响应
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
```

#### <font style="color:rgba(0, 0, 0, 0.87);">规划步骤</font>
<font style="color:rgba(0, 0, 0, 0.87);">现在让我们考虑创建规划步骤。 这将使用函数调用来创建计划。</font>

```python
from langchain_core.pydantic_v1 import BaseModel, Field


# 定义一个Plan模型类，用于描述未来要执行的计划
class Plan(BaseModel):
    """未来要执行的计划"""

    steps: List[str] = Field(
        description="需要执行的不同步骤，应该按顺序排列"
    )
```

```python
from langchain_core.prompts import ChatPromptTemplate

# 创建一个计划生成的提示模板
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """对于给定的目标，提出一个简单的逐步计划。这个计划应该包含独立的任务，如果正确执行将得出正确的答案。不要添加任何多余的步骤。最后一步的结果应该是最终答案。确保每一步都有所有必要的信息 - 不要跳过步骤。""",
        ),
        ("placeholder", "{messages}"),
    ]
)
# 使用指定的提示模板创建一个计划生成器，使用OpenAI的ChatGPT-4o模型
planner = planner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Plan)
```

```python
# 调用计划生成器，询问“当前澳大利亚公开赛冠军的家乡是哪里？”
planner.invoke(
    {
        "messages": [
            ("user", "现任澳网冠军的家乡是哪里?")
        ]
    }
)
```

```python
{'plan': ['查找2024年澳大利亚网球公开赛的冠军是谁', '查找该冠军的家乡是哪里', '用中文回答该冠军的家乡']}
```

<font style="color:rgba(0, 0, 0, 0.87);"></font>

#### <font style="color:rgba(0, 0, 0, 0.87);">重新规划步骤</font>
<font style="color:rgba(0, 0, 0, 0.87);">现在，让我们创建一个根据上一步结果重新制定计划的步骤。</font>

```python
from typing import Union


# 定义一个响应模型类，用于描述用户的响应
class Response(BaseModel):
    """用户响应"""

    response: str


# 定义一个行为模型类，用于描述要执行的行为
class Act(BaseModel):
    """要执行的行为"""

    action: Union[Response, Plan] = Field(
        description="要执行的行为。如果要回应用户，使用Response。如果需要进一步使用工具获取答案，使用Plan。"
    )


# 创建一个重新计划的提示模板
replanner_prompt = ChatPromptTemplate.from_template(
    """对于给定的目标，提出一个简单的逐步计划。这个计划应该包含独立的任务，如果正确执行将得出正确的答案。不要添加任何多余的步骤。最后一步的结果应该是最终答案。确保每一步都有所有必要的信息 - 不要跳过步骤。

你的目标是：
{input}

你的原计划是：
{plan}

你目前已完成的步骤是：
{past_steps}

相应地更新你的计划。如果不需要更多步骤并且可以返回给用户，那么就这样回应。如果需要，填写计划。只添加仍然需要完成的步骤。不要返回已完成的步骤作为计划的一部分。"""
)

# 使用指定的提示模板创建一个重新计划生成器，使用OpenAI的ChatGPT-4o模型
replanner = replanner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Act)
```

#### <font style="color:rgba(0, 0, 0, 0.87);">创建图</font>
<font style="color:rgba(0, 0, 0, 0.87);">现在我们可以创建图了！</font>

```python
from typing import Literal


# 定义一个异步主函数
async def main():
    # 定义一个异步函数，用于执行步骤
    async def execute_step(state: PlanExecute):
        plan = state["plan"]
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        task_formatted = f"""对于以下计划：
{plan_str}\n\n你的任务是执行第{1}步，{task}。"""
        agent_response = await agent_executor.ainvoke(
            {"messages": [("user", task_formatted)]}
        )
        return {
            "past_steps": state["past_steps"] + [(task, agent_response["messages"][-1].content)],
        }

    # 定义一个异步函数，用于生成计划步骤
    async def plan_step(state: PlanExecute):
        plan = await planner.ainvoke({"messages": [("user", state["input"])]})
        return {"plan": plan.steps}

    # 定义一个异步函数，用于重新计划步骤
    async def replan_step(state: PlanExecute):
        output = await replanner.ainvoke(state)
        if isinstance(output.action, Response):
            return {"response": output.action.response}
        else:
            return {"plan": output.action.steps}

    # 定义一个函数，用于判断是否结束
    def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
        if "response" in state and state["response"]:
            return "__end__"
        else:
            return "agent"
```

```python
from langgraph.graph import StateGraph, START

# 创建一个状态图，初始化PlanExecute
workflow = StateGraph(PlanExecute)

# 添加计划节点
workflow.add_node("planner", plan_step)

# 添加执行步骤节点
workflow.add_node("agent", execute_step)

# 添加重新计划节点
workflow.add_node("replan", replan_step)

# 设置从开始到计划节点的边
workflow.add_edge(START, "planner")

# 设置从计划到代理节点的边
workflow.add_edge("planner", "agent")

# 设置从代理到重新计划节点的边
workflow.add_edge("agent", "replan")

# 添加条件边，用于判断下一步操作
workflow.add_conditional_edges(
    "replan",
    # 传入判断函数，确定下一个节点
    should_end,
)

# 编译状态图，生成LangChain可运行对象
app = workflow.compile()
```

```plain
    # 将生成的图片保存到文件
    graph_png = app.get_graph().draw_mermaid_png()
    with open("plan_execute.png", "wb") as f:
        f.write(graph_png)
```

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1724582939404-0dc86760-5031-4d42-b758-76645090fc5f.png)

```python
# 设置配置，递归限制为50
config = {"recursion_limit": 50}
# 输入数据
inputs = {"input": "2024年巴黎奥运会100米自由泳决赛冠军的家乡是哪里?请用中文答复"}
# 异步执行状态图，输出结果
async for event in app.astream(inputs, config=config):
    for k, v in event.items():
        if k != "__end__":
            print(v)
```

```python
{'plan': ['查找2024年巴黎奥运会100米自由泳决赛冠军的名字', '查找该冠军的家乡']}
{'past_steps': [('查找2024年巴黎奥运会100米自由泳决赛冠军的名字', '2024年巴黎奥运会男子100米自由泳决赛的冠军是中国选手潘展乐（Zhanle Pan）。')]}
{'plan': ['查找潘展乐的家乡']}
{'past_steps': [('查找2024年巴黎奥运会100米自由泳决赛冠军的名字', '2024年巴黎奥运会男子100米自由泳决赛的冠军是中国选手潘展乐（Zhanle Pan）。'), ('查找潘展乐的家乡', '潘展乐的家乡是浙江温州。')]}
{'response': '2024年巴黎奥运会100米自由泳决赛冠军潘展乐的家乡是浙江温州。'}
```




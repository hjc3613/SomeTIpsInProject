
'''
pip install langchain
pip install langchain-community
'''

from langchain_core.tools import tool
import requests
from langchain_community.chat_models.tongyi import ChatTongyi

from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor

llm = ChatTongyi(
    model_name='qwen-turbo',api_key='sk-39204ca0f48f4279a80846112795ed63'
)

@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def divide(a: int, b:int ) -> float:
    """divide a with b, given the result of a / b

    Arges:
        a: the dividend, int
        b: the divisor, int
    """
    return a/b

@tool
def get_weather(location: str) -> str:
    """get the weather of the {location}

    Args:
        location: The address used to check the weather.
    """
    res = requests.get(f'https://restapi.amap.com/v3/weather/weatherInfo?key=2654e2fa035be16e9a495bee25cf437e&city={location}')
    return res.json()

chat_history = []

def query(question):
    global chat_history
    tools = [add, multiply, divide, get_weather]

    llm_with_tools = llm.bind_tools(tools)
    llm_with_tools
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are very powerful assistant, and you are good at using the existing tools given for you",
            ),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            "chat_history":lambda x:x['chat_history']
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    # print(list(agent_executor.stream({"input": "2345234+432345再除以5等于几？"})))
    result = agent_executor.invoke({"input": question, "chat_history":chat_history})
    chat_history.append(result['output'])
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]
    return result['output']

if __name__ == '__main__':
    while True:
        text = input("请输入问题: ")
        print(query(text))

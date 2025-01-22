import json
import traceback

from model_configurations import get_model_configuration
#from prompt_template_demo import get_prompt_template_demo_string_by_format

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

#test for fun
from langchain.prompts import ChatPromptTemplate


import requests
#hw1
from langchain.output_parsers import (ResponseSchema, StructuredOutputParser)


from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
#hw2
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
#hw3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
#hw4
import base64
from mimetypes import guess_type


gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def get_llm():
    return AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    
def demo(question):
    llm = get_llm()
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response

#print(demo("你好，使用繁體中文").content)
#print(demo("你好，使用繁體中文"))


# Test for fun
test_CPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一位擅長{talent}的女性"),
        ("human", "嗨，今天天氣不錯，有什麼{name}的新鮮事嗎?"),

    ]
)

test_prompt = test_CPT.format(talent="開玩笑", name="川普")

def testforfun001():
    llm = get_llm()
    answner = llm.invoke(test_prompt).content
    return answner

#print(testforfun001())


# hw1 practice
llm = get_llm()

response_sch_test = [
    ResponseSchema(
        name="Youtuber",
        description="訂閱數有超過10萬的youtuber",
        type="list"
    ),
    ResponseSchema(
        name="test2",
        description="訂閱數超過100萬的youtuber",
        type="用驚嘆號隔開"
    )
]

output_parser_test = StructuredOutputParser(response_schemas=response_sch_test)
format_ins = output_parser_test.get_format_instructions()
prompt_hw1_test = ChatPromptTemplate.from_messages(
    [
        ("system", "使用台灣語言並回答問題, {format_ins_input}"),
        ("human", "{question}")
    ]
)
prompt_hw1_test_input = prompt_hw1_test.partial(format_ins_input=format_ins)
reponse_hw1_output = llm.invoke(prompt_hw1_test_input.format(question="2020年台灣")).content
#print(reponse_hw1_output)


# test_123
def get_calendarific_data(year:int, month:int) -> str:
    api_key = "jh1uea8ZamUFPpYa3UynZQX96rcBmlTV"
    url = f"https://calendarific.com/api/v2/holidays?&api_key={api_key}&country=tw&year={year}&month={month}"
    response = requests.get(url)
    response = response.json()
    response = response.get('response')
    return response

class GetCalendarific(BaseModel):
    year:int = Field(description="year")
    month:int = Field(description="month")

def get_agent_executor():
    tool = StructuredTool.from_function(
        name="get_calendarific",
        description="get holidays by calendarific API key",
        func=get_calendarific_data,
        args_schema=GetCalendarific
    )
    prompt = hub.pull("hwchase17/openai-functions-agent")
    print("************************** hub.pull ****************************")
    print(prompt.messages)
    tools = [tool]
    agent = create_openai_functions_agent(get_llm(), tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    return agent_executor


def question_to_prompt(question):
    examples = [
        {"input":"2024年台灣7月紀念日有哪些?", "output":"""
        {
            "Result": [
                {
                    "date": "2024-07-01",
                    "name": "醫師節"
                },
                {
                    "date": "2024-07-15",
                    "name": "解嚴紀念日"
                }
            ]
        }
         """},
        {"input":"2024年台灣8月紀念日有哪些?", "output":"""
        {
            "Result": [
                {
                    "date": "2024-08-08",
                    "name": "父親節"
                }
            ]
        }
         """},
        {"input":"2024年台灣9月紀念日有哪些?", "output":"""
        {
            "Result": [
                {
                    "date": "2024-09-03",
                    "name": "軍人節"
                },
                {
                    "date": "2024-09-28",
                    "name": "教師節"
                }
            ]
        }
         """},
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

    responseSchema = [
        ResponseSchema(name="date", description="該紀念日的日期", type="YYYY-MM-DD"),
        ResponseSchema(name="name", description="該紀念日的名稱")
    ]
    output_parser = StructuredOutputParser(response_schemas=responseSchema)
    format_instructions = output_parser.get_format_instructions()
    format_instructions = """
    The output should be a markdown code snippet formatted in the following schema, without the leading and trailing "```json" and "```":
    {
        "Result": [
            {
                "date": YYYY-MM-DD  // 該紀念日的日期
                "name": string  // 該紀念日的名稱
            }
        ]
    }
    """
    print("************************** format_instructions ****************************")
    print(format_instructions)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "使用台灣語言並回答問題,{format_instructions1}"),
        few_shot_prompt,
        ("human", "{question1}")
    ])
    prompt = prompt.partial(format_instructions1=format_instructions)
    prompt_str = prompt.format(question1=question)
    print("************************** prompt_str ****************************")
    print(prompt_str)
    return prompt_str



def question3_to_prompt(question):
    examples = [
        {"input":"""根據先前的節日清單，這個節日{"date": "10-31", "name": "蔣公誕辰紀念日"}是否有在該月份清單？""", "output":"""
        {
            "Result": 
                {
                    "add": true,
                    "reason": "蔣中正誕辰紀念日並未包含在十月的節日清單中。目前十月的現有節日包括國慶日、重陽節、華僑節、台灣光復節和萬聖節。因此，如果該日被認定為節日，應該將其新增至清單中。"
                }
        }
         """},
        {"input":"""根據先前的節日清單，這個節日{"date": "10-10", "name": "國慶日"}是否有在該月份清單？""", "output":"""
        {
            "Result": 
                {
                    "add": false,
                    "reason": "國慶日包含在十月的節日清單中。目前十月的現有節日包括國慶日、重陽節、華僑節、台灣光復節和萬聖節。"
                }
        }
         """},
        {"input":"""根據先前的節日清單，這個節日{"date": "10-21", "name": "華僑節"}是否有在該月份清單？""", "output":"""
        {
            "Result": 
                {
                    "add": false,
                    "reason": "華僑節包含在十月的節日清單中。目前十月的現有節日包括國慶日、重陽節、華僑節、台灣光復節和萬聖節。"
                }
        }
         """},
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

    responseSchema = [
        ResponseSchema(name="add", description="""這是一個布林值，表示是否需要將節日新增到節日清單中。根據問題判斷該節日是否存在於清單中，如果不存在，則為 true；否則為 false。""", type="bool"),
        ResponseSchema(name="reson", description="描述為什麼需要或不需要新增節日，具體說明是否該節日已經存在於清單中，以及當前清單的內容。")
    ]
    output_parser = StructuredOutputParser(response_schemas=responseSchema)
    format_instructions = output_parser.get_format_instructions()
    format_instructions = """
    The output should be a markdown code snippet formatted in the following schema, without the leading and trailing "```json" and "```":
    {
        "Result":
            {
                "add": bool  // 這是一個布林值，表示是否需要將節日新增到節日清單中。根據問題判斷該節日是否存在於清單中，如果不存在，則為 true；否則為 false。
                "reson": string  // 描述為什麼需要或不需要新增節日，具體說明是否該節日已經存在於清單中，以及當前清單的內容。
            }
    }
    """
    print("************************** format_instructions ****************************")
    print(format_instructions)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "使用台灣語言並回答問題,{format_instructions1}"),
        few_shot_prompt,
        ("human", "{question1}")
    ])
    prompt = prompt.partial(format_instructions1=format_instructions)
    prompt_str = prompt.format(question1=question)
    print("************************** prompt_str ****************************")
    print(prompt_str)
    return prompt_str


def question4_to_prompt(question, data_url):
    examples = [
        {"input":"""請問中華台北的積分是多少""", "output":"""{
    "Result": 
        {
            "score": 5498
        }
}"""
         },
        {"input":"""請問日本的積分是多少""", "output":"""{
    "Result": 
        {
            "score": 6858
        }
}"""
         },
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

    responseSchema = [
        ResponseSchema(name="score", description="隊伍積分。")
    ]
    output_parser = StructuredOutputParser(response_schemas=responseSchema)
    format_instructions = output_parser.get_format_instructions()
    format_instructions = """
The output should be a markdown code snippet formatted in the following schema, without the leading and trailing "```json" and "```":
{
    "Result":
        {
            "score": int  // 隊伍積分。
        }
}
    """
    # print("************************** format_instructions ****************************")
    # print(format_instructions)
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "使用台灣語言並回答問題,{format_instructions1}"),
    #     few_shot_prompt,
    #     ("human", "{question1}")
    # ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "辨識圖片中的文字表格,{format_instructions1}"),
        few_shot_prompt,
        ("user",
            [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            ],
        ),
        ("human", "{question1}")
        ]
    )
    prompt = prompt.partial(format_instructions1=format_instructions)
    prompt_str = prompt.format_messages(question1=question)
    # print("************************** prompt_str ****************************")
    # print(prompt_str)
    return prompt_str

def get_image_url():
    # Function to encode a local image into data URL 
    def local_image_to_data_url(image_path):
        # Guess the MIME type of the image based on the file extension
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'  # Default MIME type if none is found

        # Read and encode the image file
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"

    image_path = 'baseball.png'
    data_url = local_image_to_data_url(image_path)
    return data_url











# formal release

def generate_hw01(question):
    prompt_str = question_to_prompt(question)
    return demo(prompt_str).content
    
def generate_hw02(question):
    agent_executor = get_agent_executor()
    prompt_str = question_to_prompt(question)
    response = agent_executor.invoke({"input": prompt_str}).get('output')
    print("************************** response ****************************")
    print(response)
    return response
    
def generate_hw03(question2, question3):
    agent_executor = get_agent_executor()
    prompt_str = question_to_prompt(question2)
    prompt_str3 = question3_to_prompt(question3)
    history = ChatMessageHistory()
    def get_history() -> ChatMessageHistory:
        return history
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    print("************************** history.messages ****************************")
    print(history.messages)
    response = agent_with_chat_history.invoke({"input":prompt_str}).get('output')
    print("************************** history.messages ****************************")
    print(history.messages)
    response = agent_with_chat_history.invoke({"input":prompt_str3}).get('output')
    print("************************** history.messages ****************************")
    print(history.messages)

    print("************************** response ****************************")
    return response
    
def generate_hw04(question):
    llm = get_llm()
    data_url = get_image_url()
    prompt_str = question4_to_prompt(question, data_url)
    response = llm.invoke(prompt_str).content
    print("************************** response ****************************")
    return response





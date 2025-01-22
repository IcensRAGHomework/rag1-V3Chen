import json
import traceback

from model_configurations import get_model_configuration
#from prompt_template_demo import get_prompt_template_demo_string_by_format

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

#test for fun
from langchain.prompts import ChatPromptTemplate

#hw1
from langchain.output_parsers import (ResponseSchema, StructuredOutputParser)

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def generate_hw01(question):
    pass
    
def generate_hw02(question):
    pass
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
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
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    answner = llm.invoke(test_prompt).content
    return answner

#print(testforfun001())


# hw1 practice
llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

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
print(reponse_hw1_output)



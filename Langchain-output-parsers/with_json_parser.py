from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    model='openai/gpt-oss-120b',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template1 = PromptTemplate(
    template='Give me full name, city and age of {person} \n, {format}',
    input_variables=['person'],
    partial_variables={'format':parser.get_format_instructions()}
)

chain = template1 | model | parser

result = chain.invoke({'person':'Rohit sharma'})

print(result)
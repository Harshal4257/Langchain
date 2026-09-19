from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template='Give a detailed summary on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Tell me 5 important points about {text}',
    input_variables=['text']
)

llm = HuggingFaceEndpoint(
    model='openai/gpt-oss-120b',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

parser =  StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'Cricket'})

print(result)
chain.get_graph().print_ascii()
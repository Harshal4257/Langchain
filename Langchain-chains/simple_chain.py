from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    model='openai/gpt-oss-120b',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template='Give me 5 interesting facts about {topic}',
    input_variables=['topic']
)

parser =  StrOutputParser()

chain = template | model | parser

result = chain.invoke({'topic':'cricket'})

print(result)
chain.get_graph().print_ascii()  
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    model='openai/gpt-oss-120b',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

class review(BaseModel):
    name : str = Field(description='Name of the person')
    age : int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=review)

template =  PromptTemplate(
    template='Give me name , age and city of {region} person \n {format}',
    input_variables=['region'],
    partial_variables={'format':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'region':'Indian'})
print(result)
print(template)

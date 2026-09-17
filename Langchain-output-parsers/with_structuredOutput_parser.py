from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers.structured import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    model='openai/gpt-oss-120b',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='Face_1', description='Face 1 about the topic'),
    ResponseSchema(name='Face_2', description='Face 2 about the topic'),
    ResponseSchema(name='Face_3', description='Face 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template1 = PromptTemplate(
    template='Give me 3 facts about {topic} \n, {format}',
    input_variables=['topic'],
    partial_variables={'format':parser.get_format_instructions()}
)

chain = template1 | model | parser

result = chain.invoke({'topic':'India'})

print(result)
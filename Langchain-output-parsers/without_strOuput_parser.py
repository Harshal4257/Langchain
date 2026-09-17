from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    model='openai/gpt-oss-120b',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Write 5 line summnary on following topic. \n {text}',
    input_variables=['text']
)

promt1 = template1.invoke({'topic':'Black Hole'})

result = model.invoke(promt1)

prompt2 = template2.invoke({'text':result.content})

result2 = model.invoke(prompt2)

print(result2.content)

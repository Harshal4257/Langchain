from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    model="openai/gpt-oss-120b",
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about langchain')
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)
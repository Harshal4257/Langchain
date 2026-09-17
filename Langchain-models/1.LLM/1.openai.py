from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model='gpt-4')

result = llm.invoke('Who is prime minister of india')
print(result)

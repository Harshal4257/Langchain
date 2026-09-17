from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpointEmbeddings(
    repo_id='sentence-transformers/all-MiniLM-L6-v2'
)

result = llm.embed_query('Delhi is the capital city of india')
print(str(result))

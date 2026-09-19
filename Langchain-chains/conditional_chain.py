import warnings
from transformers.utils import logging

warnings.filterwarnings("ignore")

logging.set_verbosity_error()
logging.disable_progress_bar()

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
import torch
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    model="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment : Literal['positive','negative'] = Field(description='Sentiment of the given feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of following feedback into postive or negative {feedback} \n {format_instructions}",
    input_variables=['feedback'],
    partial_variables={'format_instructions' : parser2.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template='Write an appropriate 3 lines response for this positve feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate 3 lines response for this negative feedback \n {feedback}',
    input_variables=['feedback']
)

sentiment_chain = prompt1 | model | parser2

branch_chain = RunnableBranch(
    (lambda x : x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x : x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: 'could not find sentiment')
)

chain = sentiment_chain | branch_chain

print(chain.invoke({'feedback':'This phone is wonderful'}))







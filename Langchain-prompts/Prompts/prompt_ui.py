from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    model="openai/gpt-oss-120b",
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

st.header("Research Assistant Tool")
paper_input = st.selectbox('Select the paper',['Attention is all you need','Language Models are Unsupervised Multitask Learners',''
             'Neural Machine Translation by Jointly Learning to Align and Translate'])
style_input = st.selectbox('Select explanation style',['Beginner','Tecnhical','Code-oreinted','Mathematical'])
length_input = st.selectbox('Select explanation length',['Short(1-2 paraghaphs)','medium(2-3 paragraphs)','long(4-5 paragraphs)'])

template = load_prompt('template.json')

prompt = template.invoke(
    {'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input}
)

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)
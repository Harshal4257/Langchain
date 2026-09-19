import warnings
from transformers.utils import logging

warnings.filterwarnings("ignore")

logging.set_verbosity_error()
logging.disable_progress_bar()

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import torch

load_dotenv()

prompt1 = PromptTemplate(
    template="Give me 5 lines summary about {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Create 3 quentions and answer on this {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="merge this {notes} and this {quiz} and show them one after another",
    input_variables=['notes','quiz']
)

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",

    device=0,  # NVIDIA GPU

    model_kwargs={
        "torch_dtype": torch.float16
    },

    pipeline_kwargs={
        "max_new_tokens": 100,
        "do_sample": False
    }
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

runnable = RunnableParallel({
    'notes': prompt1 | model | parser,
    'quiz': prompt2 | model | parser
})

merge_chain = prompt3 | model | parser

chain = runnable | merge_chain | parser

text = """
Machine learning (ML) is a subset of artificial intelligence (AI) focused on building systems that learn from data to improve their performance without being explicitly programmed. Instead of relying on rigid, human-written rules, machine learning algorithms analyze large datasets, identify underlying statistical patterns, and use those insights to make autonomous decisions or predictions.How Machine Learning WorksThe standard machine learning workflow bridges the gap between raw data and actionable predictions through five main steps:Data Collection: Gathering relevant information such as text, images, or numbers.Algorithm Selection: Choosing an appropriate mathematical model based on the goal (e.g., classifying images or forecasting prices).Training: Feeding the data into the algorithm so it can recognize patterns and adjust itself to minimize errors.Evaluation: Testing the trained model on unseen data to ensure it can generalize accurately to real-world scenarios.Prediction: Deploying the model to process new, incoming data and generate outputs automatically.The Core Types of Machine LearningMachine learning approaches vary depending on how the algorithm receives feedback during training:TypeHow It WorksCommon Use CasesSupervised LearningTrain models using labeled datasets where the correct answers are already provided.Spam filtering, credit score estimation, and image classification.Unsupervised LearningFinds hidden structures or groupings in unlabeled data without human guidance.Customer market segmentation and anomaly detection.Reinforcement LearningAn agent learns by trial and error, receiving rewards or penalties based on its actions.Robotics, autonomous driving, and video game AI.Generative AILearns the underlying distribution of a dataset to create entirely new content.Writing text, creating artwork, or generating music.Real-World ApplicationsMachine learning powers many of the digital tools used daily:Recommendation Engines: Platforms like Netflix and Spotify analyze user preferences to suggest movies and music.Fraud Detection: Banks monitor transaction history in real-time to instantly flag suspicious behavior.Natural Language Processing: Smart assistants and customer chatbots translate languages and interpret voice commands.
"""

result = chain.invoke({'text': text})

#print(result)

chain.get_graph().print_ascii()



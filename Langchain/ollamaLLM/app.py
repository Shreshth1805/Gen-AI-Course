import os
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

## prompt Template:
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful assistant. Please respond to the que asked"),
        ("user","Question:{question}")
    ]
)

## Streamlit framework:
st.title("Ollama Model")
input_text=st.text_input("What question you have in your mind?")

## Ollama llama3 model:
llm = OllamaLLM(model="llama3")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))
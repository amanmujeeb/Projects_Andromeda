#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 12:12:41 2023

@author: amanmujeeb
"""

import streamlit as st
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import pandas as pd

def generate_response(uploaded_file, openai_api_key, query_text, chat_history):
    # Load document if file is uploaded
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        qa = create_pandas_dataframe_agent(OpenAI(openai_api_key=openai_api_key), 
                                         df, 
                                         verbose=True)
        return qa.run(query_text, chat_history=chat_history)

# Page title
st.set_page_config(page_title='Ask the Csv App')
st.title('Ask the Csv App')

# File upload
uploaded_file = st.file_uploader('Upload a Csv', type='csv')
#logo
logo = "andromeda.jpeg"  # Replace with the actual filename of your logo
st.sidebar.image(logo, use_column_width=True)
# Query text
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_file)

# Chat history
chat_history = st.empty()
if st.session_state.get('chat_history') is not None:
    chat_history.markdown('\n\n'.join(st.session_state['chat_history']))

# Form input and query
result = []

with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text, st.session_state.get('chat_history', []))
            result.append(response)
            st.session_state['chat_history'] = st.session_state.get('chat_history', []) + [f"**User**: {query_text}", f"**Bot**: {response}"]
            del openai_api_key

if len(result):
    st.info(response)
    
# Save chat history
if 'chat_history' in st.session_state:
    st.session_state['chat_history'] = st.session_state['chat_history'][-20:]  # Limit the chat history length

# Display chat history
chat_history.markdown('\n\n'.join(st.session_state.get('chat_history', [])))

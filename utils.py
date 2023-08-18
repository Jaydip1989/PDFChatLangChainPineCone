from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from config import apikey, pinecone_apikey
import openai
import streamlit as st

embeddings = OpenAIEmbeddings(
    model_name = "ada", openai_api_key=apikey
)

pinecone.init(      
	api_key= pinecone_apikey,      
	environment='us-west1-gcp-free'      
)      
index = pinecone.Index('chatbot')

def find_match(input):
    query_result = embeddings.embed_query(input)
    result = index.query(query_result, top_k=2, include_metadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):
    response = openai.Completion.create(
        model = 'text-davinci-003',
        prompt = f"Given the following user query and conversation log, formulate a question that would \
            be the most relevant to provide the user with an answer from the knowledge base. \
            \n\n CONVERSATION LOG: \n {conversation} \n\n Query: {query} \n\n Refined Query: ",
        temperature = 0.7,
        max_tokens = 256,
        top_p = 1,
        frequency_penalty = 0,
        presence_penalty = 0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ''
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i] + "\n"
    return conversation_string
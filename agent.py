import os
import json

import streamlit as st

from llm_client import LLMClient
from rag_builder import RAGBuilder

def init_llm_client():
   return LLMClient()

def start_streamlit_session():
    """
    Initializes Streamlit session.
    """
    retriever = None
    st.title("Thoughtful AI Agent")
    st.subheader("We are here to answer your Thoughtful and healthcare questions")

    if "messages" not in st.session_state:
        st.session_state.messages=[]
    else: #retrieve messages from session and show it
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    
    if "llmclient" not in st.session_state:
        st.session_state.llmclient = init_llm_client()
        rag_builder = RAGBuilder(st.session_state.llmclient)
        #parameterize
        st.session_state.retriever = rag_builder.get_rag_retriever("./data/sample.json")


    #answer questions
    if prompt := st.chat_input("What do you want to know?"):
        #save to historu
        st.session_state.messages.append({"role":"user", "content": prompt})

        with st.chat_message("user"): #show prompted question
            st.markdown(prompt)
        
        #use the retriever to find similarity search or max relevance search
        #k=1 one answer enough?
        answers = st.session_state.retriever.vectorstore.max_marginal_relevance_search(prompt, k=1)
        # print(answers)
        context = ""
        for answer in answers:
            #use answer as primary content?
            context = context + f"; Content: {answer.metadata.get("answer")} \n" + f" Related question: {answer.page_content}" 

        with st.chat_message("agent"):
            message_placeholder = st.empty()
            text = ""
            #use llm client to get chat completion response
            text_stream = st.session_state.llmclient.get_stream(st.session_state.messages)
            for chunk in text_stream:
                content = chunk.choices[0].delta.content
                if content is not None: text += content #accumalate response
                message_placeholder.markdown(text)
            message_placeholder.markdown(text)

        #save agent answer to history
        st.session_state.messages.append({"role":"agent", "content": text})

#main code
if __name__ == "__main__":
    start_streamlit_session()
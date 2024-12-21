import os
import json

import streamlit as st

from llm_client import LLMClient

def init_llm_client():
   return LLMClient()

def start_streamlit_session():
    """
    Initializes Streamlit session.
    """
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


    #answer questions
    if prompt := st.chat_input("What do you want to know?"):
        #save to historu
        st.session_state.messages.append({"role":"user", "content": prompt})

        with st.chat_message("user"): #show prompted question
            st.markdown(prompt)
        
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
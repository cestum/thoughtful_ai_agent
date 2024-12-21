import os
import json

import streamlit as st

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

    #answer questions
    if prompt := st.chat_input("What do you want to know?"):
        #save to historu
        st.session_state.messages.append({"role":"user", "content": prompt})

        with st.chat_message("user"): #show prompted question
            st.markdown(prompt)
        
        with st.chat_message("agent"): #just echo for now
            st.markdown(prompt)
        #save agent answer to history
        st.session_state.messages.append({"role":"agent", "content": prompt})

#main code
if __name__ == "__main__":
    start_streamlit_session()
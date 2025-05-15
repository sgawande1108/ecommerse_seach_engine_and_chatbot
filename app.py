
import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"  # or use "none"




import streamlit as st
import pandas as pd
from dataset import retrieve_data
from agent2 import agent_response

st.set_page_config(page_title="College E-commerce Search", layout="wide")

st.title("🎓E-commerce Platform")

col1, col2 = st.columns([2, 3])

with col1:
    st.header("🔍 Search Products")
    #search_query = st.text_input("Enter your search query:", key="search_box")

    search_query = st.text_input("🔍 What are you looking for?", placeholder="e.g. waterproof smart watch under 3000")

    if search_query:
        with st.spinner("Finding the best matches for you..."):
            results = retrieve_data(search_query)
            st.subheader("🔎 Top Matching Products")
            for _, row in results.iterrows():
                with st.container():
                    st.markdown(f"**{row['title']}**")
                    st.write(row['description'][:150] + "...")
                    st.markdown(f"💰 **Price:** ${row['price']} &nbsp;&nbsp; ⭐ **Rating:** {row['average_rating']}")
                    st.markdown("---")


with col2:
    st.header("🤖 Chat with Agent")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    chat_input = st.text_input("Ask a question or get help:", key="chat_input", placeholder="e.g. How to track my order?")

  
    if chat_input:
      
        st.session_state.chat_history.append(("You", chat_input))

        # Call the agent and handle response/errors
        with st.spinner("Agent is thinking..."):
            try:
                response = agent_response(chat_input)

                
                if not response:
                    response = "⚠️ Agent did not return a response."

                st.session_state.chat_history.append(("Agent", response))

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.session_state.chat_history.append(("Agent", error_msg))
                st.error(error_msg)

   
    for speaker, message in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**🧑 You:** {message}")
        else:
            st.markdown(f"**🤖 Agent:** {message}")

    
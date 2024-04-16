import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import streamlit as st
from search_assistant.src.search_assistant import SearchAssistant

def handle_user_input(user_question):
    if user_question:
        with st.spinner("Processing..."):
            if st.session_state.method=="rag_query":
                response = st.session_state.search_assistant.retrieval_call(user_question)
                sources = set(f'{doc.metadata["source"]}'for doc in response["source_documents"])
                st.session_state.related_queries_history.append(
                    st.session_state.search_assistant.get_relevant_queries(
                        st.session_state.query+" - "+user_question
                        )
                    )
            elif st.session_state.method=="basic_query":
                response = st.session_state.search_assistant.basic_call(
                    query = user_question,
                    search_method = st.session_state.tool[0],
                    max_results = st.session_state.max_results,
                    search_engine = st.session_state.search_engine)
                sources = set(response["sources"])
                st.session_state.related_queries_history.append(st.session_state.search_assistant.get_relevant_queries(user_question))
        st.session_state.chat_history.append(user_question)
        st.session_state.chat_history.append(response["answer"])
 
        # Create a Markdown string with each source on a new line as a numbered list with links
        sources_text = ""
        for index, source in enumerate(sources, start=1):
            source_link = source
            sources_text += (
                f'<font size="2" color="grey">{index}. {source_link}</font>  \n'
            )
            
        st.session_state.sources_history.append(sources_text)   

    for ques, ans, source, related_queries in zip(
        st.session_state.chat_history[::2],
        st.session_state.chat_history[1::2],
        st.session_state.sources_history,
        st.session_state.related_queries_history,
    ):
        # use question
        with st.chat_message("user"):
            st.write(f"{ques}")

        #kit response
        with st.chat_message(
            "ai",
            avatar="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
        ):
            # answer
            st.write(f"{ans}")
            # sources
            if st.session_state.show_sources:
                with st.popover("Sources", use_container_width=False):
                    sources_lines = source.split("\n")[:-1]
                    for i in range(len(sources_lines)//3+1):
                        columns = st.columns(3)
                        for j in range(len(columns)):
                            if i*3+j >= len(sources_lines): 
                                break
                            columns[j].container(border=True).markdown(
                                f'<font size="2" color="grey">{sources_lines[i*3+j]}</font>',
                                unsafe_allow_html=True,
                            )
            # related questions
            with st.expander("**Related questions**", expanded=False):
                if len(related_queries)>0:
                    for question in related_queries:
                        st.markdown(
                            f"[{question}](https://www.google.com/search?q={question.replace(' ', '+')})",
                        )
             
def main():
    st.set_page_config(
        page_title="AI Starter Kit",
        page_icon="https://sambanova.ai/wp-content/uploads/2021/05/logo_icon-footer.svg",
        #layout="wide"
    )
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "sources_history" not in st.session_state:
        st.session_state.sources_history = []
    if "related_queries_history" not in st.session_state:
        st.session_state.related_queries_history = []
    if "related_questions" not in st.session_state:
        st.session_state.related_questions = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    if "search_assistant" not in st.session_state:
        st.session_state.search_assistant = None
    if "tool" not in st.session_state:
        st.session_state.tool = "serpapi"
    if "search_engine" not in st.session_state:
        st.session_state.search_engine="google"
    if "max_results" not in st.session_state:
        st.session_state.max_results = 5
    if "method" not in st.session_state:
        st.session_state.method = "basic_query"
    if "query" not in st.session_state:   
        st.session_state.query = None
    if "input_disabled" not in st.session_state:
        st.session_state.input_disabled = True
    
    st.title(":orange[SambaNova] Search Assistant")
    
    with st.sidebar:
        st.title("**Setup**")
        
        tool=st.radio("Select Search Tool to use", ["serpapi","serper","openserp"])
        if tool == "serpapi":
            st.session_state.tool = ["serpapi"]
            st.session_state.search_engine=st.selectbox("Search engine to use", ["google","bing"])
        elif tool == "serper":
            st.session_state.tool = ["serper"]
            st.session_state.search_engine=st.selectbox("Search engine to use", ["google"])
        elif tool == "openserp":
            st.session_state.tool = ["openserp"]
            st.session_state.search_engine=st.selectbox("Search engine to use", ["google","baidu"])
            
        st.session_state.max_results=st.slider("Max number of results to retrieve", 1, 20, 5)
        
        st.markdown("Method for retrieval")
        method = st.selectbox("Method for retrieval", ["Search and answer","Search and scrape sites"])
        if method=="Search and scrape sites":
            st.session_state.query = st.text_input("Query")
            
        if st.button("set"):
            st.session_state.search_assistant=SearchAssistant()
            with st.spinner("setting searchAssistant" if method=="Search and answer" else "searching and scraping sites"):
                if method=="Search and scrape sites":
                    st.session_state.method="rag_query"
                    if not st.session_state.query:
                        st.error("Please enter a query")
                    else:
                        st.session_state.search_assistant.search_and_scrape(
                            query=st.session_state.query,
                            search_method=st.session_state.tool[0],
                            max_results=st.session_state.max_results,
                            search_engine=st.session_state.search_engine)
                        st.session_state.input_disabled=False
                        st.toast("Search done and knowledge base updated you can chat now")
                elif method=="Search and answer":
                    st.session_state.method=="basic_query"
                    st.session_state.input_disabled=False
                    st.toast("Settings updated you can chat now")
        if  st.session_state.search_assistant:
            if  st.session_state.search_assistant.urls:                       
                with st.expander("Scraped sites", expanded = True):
                    st.write(st.session_state.search_assistant.urls)
                    
        with st.expander("Additional settings", expanded=True):
            st.markdown("**Interaction options**")
            st.markdown(
                "**Note:** Toggle these at any time to change your interaction experience"
            )
            st.session_state.show_sources = st.checkbox("Show sources", value=True)

            st.markdown("**Reset chat**")
            st.markdown(
                "**Note:** Resetting the chat will clear all conversation history and not updated documents."
            )
            if st.button("Reset conversation"):
                st.session_state.chat_history = []
                st.session_state.sources_history = []
                st.session_state.related_queries_history = []
                st.rerun()
            
    user_question = st.chat_input("Ask questions about data in provided sites", disabled=st.session_state.input_disabled)
    handle_user_input(user_question)

if __name__ == '__main__':
    main()
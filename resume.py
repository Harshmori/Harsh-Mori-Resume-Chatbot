import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as ps
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

def load_env_variables():
    """Load environment variables from .env file."""
    load_dotenv()

def check_api_keys():
    """Check if API keys are set."""
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

def setup_page():
    """Setup Streamlit page configuration."""
    st.set_page_config(
        page_title="Your own ChatGPT",
        page_icon="📖"
    )

def init():
    """Initialize the environment and page settings."""
    load_env_variables()
    check_api_keys()
    setup_page()

def retrieve_query(index, query, k=2):
    """Retrieve matching results from the Pinecone index for a given query."""
    if not isinstance(query, str):
        raise TypeError("Query must be a string")
    matching_results = index.similarity_search(query, k=k)
    return matching_results

def retrieve_answers(chain, index, query):
    """Retrieve answers from the chain based on the query."""
    text = "state chapter number and verse number along with the consice answer ONLY IF the question is related to bhagawad gita"
    doc_search = retrieve_query(index, query)
    print(doc_search)
    response = chain.run(input_documents=doc_search, question=query)
    return response

def main():
    """Main function to run the chatbot application."""
    init()

    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")

    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
    os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
    index_name = 'vartalap'
    index = ps.from_existing_index(index_name=index_name, embedding=embeddings, namespace="res")


    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant of harsh mori, you task will be to give consice and helpful answers to the questions being asked related to harsh mori"),
            HumanMessage(content="What can i ask this bot ?"),
            AIMessage(content='''
Try asking the following questions:
                      
📞 List out your Contact Details
🎓 Where have you got your education from ?
💼 Talk about your work experiences
💻 List out projects you have done along with their links''')
        ]

    st.header("Harsh Mori Resume Chatbot")
    with st.sidebar:
        st.info('''# Harsh Mori's Resume Chatbot

Welcome to the Harsh Mori Resume Chatbot! 🤖 This is a personal assistant dedicated to providing you with all the details you need! 🌟

You can chat with the bot to:

- 📞 Get **Contact Details**
- 🎓 Learn about **Education**
- 💼 Explore **Work Experience**
- 💻 Dive into **Projects**
- 🛠️ Discover **Skills**

Ask away, and let the Harsh Mori Resume Chatbot assist you in uncovering all the fascinating aspects of Harsh's professional journey! 🚀
''')
        st.markdown("### Made By [Harsh Mori](https://github.com/Harshmori)")

    # Sidebar with user input
    user_input = st.chat_input("Your question: ", key="user_input")

    # Handle user input
    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Thinking..."):
            # Only pass the text of the latest human message to the retrieve_answers function
            query = st.session_state.messages[-1].content
            response = retrieve_answers(chain, index, query)
        st.session_state.messages.append(AIMessage(content=response))
    
    # Display message history
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')

if __name__ == '__main__':
    main()

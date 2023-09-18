# from langchain.llms import OpenA
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools, Tool
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.error("Please add your OpenAI API key to the sidebar.")
    st.stop()

    
llm = ChatOpenAI(temperature=0, streaming=True, openai_api_key=openai_api_key)
search = DuckDuckGoSearchAPIWrapper()
tools = [
  *load_tools(["ddg-search", "llm-math", "wikipedia", "arxiv"], llm=llm),
  Tool(
    name="Intermediate Answer",
    func=search.run,
    description="useful for when you need to ask with search",
  ),
]

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)
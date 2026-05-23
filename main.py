from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

#1 Commented to use Tavily Search tool inplace of custom too `search` written by me.
#1 from tavily import TavilyClient

#1 tavily = TavilyClient()

#1 @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query: The query to search for
    
#     Returns:
#         The search result
#     """

#     print(f"called tool search: Searching for {query}")
#     return tavily.search(query=query)


llm = ChatOllama(temperature=0, model="llama3.2")
tools = [TavilySearch]  #1 [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    # result = agent.invoke({"messages": HumanMessage(content="what is the weather in Delhi")})
    result = agent.invoke({"messages": HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the Bangalore location India on linkedin and list their details")})
    print(result)


if __name__ == "__main__":
    main()

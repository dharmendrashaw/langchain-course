from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

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

class Source(BaseModel):
    """Schema for source used by the agent"""
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the agent response with answer and source"""
    
    answer: str = Field(description="The agent's response with answer")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")


llm = ChatOllama(temperature=0, model="qwen3.5:2b") 
tools = [TavilySearch()]  #1 [search]
# Response format mostly doesn't work with small model with AgentResponse. 
# need to test on bigger mode like 8B or greater.
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    # result = agent.invoke({"messages": HumanMessage(content="what is the weather in Delhi")})
    result = agent.invoke({"messages": HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the Bangalore location India on linkedin and list their details")})
    print(result)


if __name__ == "__main__":
    main()

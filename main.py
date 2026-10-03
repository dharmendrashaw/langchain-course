import os

from operator import itemgetter
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

print("Initializing components")

embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        output_dimensionality=1536,
)

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

vectorstore = PineconeVectorStore(index_name=os.environ['INDEX_NAME'], embedding=embeddings)

retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """
Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:
"""
)

def format_doc(docs):
    """Format retrived documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

#lang chain expression language (lcel)
def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generate a response.
    
    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """

    # Step 1: Retrieve relavant documents
    docs = retriever.invoke(query)

    # Step 2: Format documents into content string
    context = format_doc(docs=docs)

    # Step 3: Format prompt with content and question
    message = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with message
    response = llm.invoke(message)

    # Step 5: Return content
    return response.content


# RAG retrival with langchain expression language
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """

    # flow - check retrieval_chain_without_lcel() to understand the flow without lcel
    # get-question -> retrive-using-question (from Vector store) -
    # -> format-docs (retreived docs from vector store) -> create-promt -> invoke-llm -> parse-output
    retrieval_chain = (
        RunnablePassthrough.assign( # RuunablePassThrough.assign will take the question as input and create a context and pass the input as is. e.g - {question:"", context: ""}
            context = itemgetter("question") | retriever | format_doc
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain



if __name__ == "__main__":
    print("Retrieving...")

    query = "What is Pinecone in machine learning"

    # Approach 1: Raw invocation without RAG
    print("\n" + "=" * 70)
    print("Approach 1: Raw LLM invocation without RAG")
    print("\n" + "=" * 70)
    # result_raw = llm.invoke([HumanMessage(content=query)])
    # print("\nAnswer")
    # print(result_raw.content)

    # Approach 2: RAG implementation without LCEL
    print("\n" + "=" * 70)
    print("Approach 2: RAG without LCEL")
    print("\n" + "=" * 70)
    # result_without_lcel = retrieval_chain_without_lcel(query=query)
    # print("\nAnswer")
    # print(result_without_lcel)

    # Approach 3: RAG implementation with LCEL
    print("\n" + "=" * 70)
    print("Approach 3: RAG with LCEL")
    print("\n" + "=" * 70)
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer")
    print(result_with_lcel)

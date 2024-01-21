
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.pinecone import Pinecone as lcpc
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
load_dotenv()


llm = ChatOpenAI()
loader = CSVLoader(file_path="test1.csv")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

print(docs)



# loader = TextLoader("123.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

# embeddings = OpenAIEmbeddings()
# print(docs)


# initialize pinecone
pc = Pinecone()

index_name = "langchain-demo7"

# First, check if our index already exists. If it doesn't, we create it
if index_name not in pc.list_indexes().names():
    # we create a new index
    pc.create_index(
    name=index_name,
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud='aws', 
        region='us-west-2'
    ) 
)
    docsearch = lcpc.from_documents(docs, embeddings, index_name=index_name)

else:
    print("Do not excist")
    docsearch = lcpc.from_existing_index(index_name, embeddings)
#
# The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`

# index = pc.Index(index_name)
def retrieve_info(query):
    similar_response = docsearch.similarity_search(query, k=3)
    page_content_array = [doc.page_content for doc in similar_response]

    print(page_content_array) 
    return page_content_array    


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
You are an crazy optimistic life coach, that answers everyhning 
with alot of happiness and alot of energy.

Below is a message that i have received from a customer:
{message}

Here is a list of answers we give our customers:
{best_practice}

Please write an answer realitaing to the best practice:
"""

prompt = PromptTemplate(
    input_variables= ["message", "best_practice"],
    template = template
)

chain = LLMChain(llm=llm, prompt=prompt)

# def generate_response(message):
#     best_practice = retrieve_info(message)
#     response = chain.invoke(message,best_practice)
#     return response

# message = """What is the weather like?"""

# response = generate_response(message)
# print(response)

retriever = docsearch.as_retriever()
rag_chain = (
    {"message": retriever | retrieve_info, "best_practice": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is the weather like?")

print(response)
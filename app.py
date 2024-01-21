
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
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

output_parser = StrOutputParser()


loader = CSVLoader(file_path="archive/all_bikez_curated_testfil_2.csv", source_column="Brand")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

pc = Pinecone()

index_name = "motorcycle-1000"

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
   
    docsearch = lcpc.from_existing_index(index_name, embeddings)
#
# The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`


prompt = ChatPromptTemplate.from_template("""
Du är en svensk mekaniker som hjälper till att svara i uppgifter om motorcyklar, vet du inte svaret så ska du säga det.
Nedanför är en fråga som en kund har skrivit
{message}

Här är en lista över motorcyklar och deras specifikationer som du hittar ditt svar på, 
sortera nummer i storleksordning:
{best_practice}

Skriv ett svar baserat på best practice:
"""
)
def retrieve_info(query):
    similar_response = docsearch.similarity_search(query, k=10)
    page_content_array = [doc.page_content for doc in similar_response]

     
    return page_content_array   


rag_chain = (
    {"best_practice": retrieve_info, "message": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("Vilken aprilia,rs 50 hade bäst rating?")
print(response)
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from vertexai.language_models import TextEmbeddingModel
import google.generativeai as genai
#from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.llms import OpenAI
import csv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain



from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials

#genai.configure(api_key='AIzaSyD7ouJHCLwniBMlcb-BGrldEeyMPc_MPEI')

key_path="F:\GCP-Devops\gcp_key.json"

credentials =Credentials.from_service_account_file(key_path , scopes=['https://www.googleapis.com/auth/cloud-platform'])

if credentials.expired:
    credentials.refresh(Request())

	
PROJECT_ID='theta-spirit-414223'
REGION = 'us-central1'
API_ENDPOINT="us-central1-aiplatform.googleapis.com"
MODEL_ID="gemini-1.0-pro-001"
LOCATION_ID="us-central1"
project_id = "theta-spirit-414223"
location = "us-central1"


import vertexai


from vertexai.language_models import TextGenerationModel
from vertexai.language_models import ChatModel, InputOutputTextPair
from vertexai.generative_models import(GenerationConfig , GenerativeModel)
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai.embeddings import VertexAIEmbeddings


vertexai.init(project=PROJECT_ID , location = REGION , credentials= credentials)


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_csv_text(csv_file):
    text = ""
    for csv in csv_file:
        with open(csv, newline="") as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
      # Combine all elements of each row into a single string
             text += ",".join(row) + "\n"  # Add newline character for each row

    return text



def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    #chunks = text_splitter.split_text(text)
    chunks = text_splitter.split_documents(documents)
    return chunks


def get_vector_store(text_chunks):
   # embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")
    #model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
   # embeddings = model.get_embeddings(text_chunks)
    
    persist_directory='db'
    vectordb= Chroma.from_documents(text_chunks,
                                    embeddings,
                                    ids = [f"{item.metadata['source']}-{index}" for index, item in enumerate(text_chunks)],
                                    persist_directory='db',)
    vectordb.persist()
    print("Total Data Abhishek Have")
    print(vectordb._collection.count())
    # = FAISS.from_documents(text_chunks, embedding=embeddings)
    #retriever = vector_store.as_retriever(search_kwargs={"k" : 2})
    #vector_store.save_local("faiss_index")

    


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = VertexAI(model_name="text-bison@001")

    memory = ConversationBufferMemory(return_messages=True , memory_key="chat_history")

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    #chain =  ConversationalRetrievalChain.from_llm(model,memory=memory,chain_type="refine", retriever=vectordb.as_retriever(), prompt=prompt)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):

    embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")
    persist_directory='db'
    new_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    docs = new_db.similarity_search(user_question,k=20)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config(
    page_title="Chat with Abhi Payment System",
    page_icon=":brain:",  # Favicon emoji
    layout="centered",  # Page layout option
)

    #st.mardown(page_bg_img,unsafe_allow_html=True)

    st.header("Chat with Abhi Payment System💁")

    user_question = st.text_input("Ask a Question about your Payment Data")

    if user_question:
       user_input(user_question)
        

    with st.sidebar:
        st.title("Menu:")
        #pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        csv_file = st.file_uploader("Upload your CSV Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                #raw_text = get_pdf_text(pdf_docs)
                #raw_text = get_csv_text(csv_file)
                loader = CSVLoader(file_path="F:\Virtusa_Gen_AI_Hackathon\sample.csv")
                documents = loader.load()
                text_chunks = get_text_chunks(documents)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()

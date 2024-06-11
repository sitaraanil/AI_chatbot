import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import langchain
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from transformers import AutoTokenizer, AutoModel
import torch
import os
import openai
 
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)

 
load_dotenv()

def generate_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Tokenize the input texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    return embeddings

def main():
    st.header("Chat with PDF 💬")
 
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    # Initialize the chat messages history
    # if "messages" not in st.session_state.keys():
    #     st.session_state.messages = [
    #         {"role": "assistant", "content": "How can I help?"}
    #     ]
    # query = st.chat_input()
    # # Prompt for user input and save
    # if prompt := query:
    #     st.session_state.messages.append({"role": "user", "content": prompt})

    # # display the existing chat messages
    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.write(message["content"])

    # st.write(pdf)

    if pdf is not None:
        with st.spinner("Please wait for the document to load."):
            pdf_reader = PdfReader(pdf)
            # query = st.text_input("Ask questions about your PDF file:")
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
    
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
                )
            chunks = text_splitter.split_text(text=text)
    
            # # embeddings
            store_name = pdf.name[:-4]
            # st.write(f'{store_name}')
            # st.write(chunks)
    
            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
                # st.write('Embeddings Loaded from the Disk')s
            else:
                embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
                # embeddings = generate_embeddings(chunks)
                # embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)
                    print("embedded: ",embeddings)
    
            # embeddings = OpenAIEmbeddings()
            # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
    
            # Accept user questions/query
            # query = st.text_input("Ask questions about your PDF file:")
            # st.write(query)
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [
                {"role": "assistant", "content": "How can I help?"}
            ]
        query = st.chat_input()
    # Prompt for user input and save
        if prompt := query:
            st.session_state.messages.append({"role": "user", "content": prompt})

        # display the existing chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if query:
                docs = VectorStore.similarity_search(query=query, k=3)
                prompt = ChatPromptTemplate.from_messages([
                                ("system", f"You are a bot that summarises policy documents. Generate responses based on the document and user queries. Do not fabricate information. Inform the user if the information is unavailable.")
                                # ("human", f""),
                            ])
                llm = OpenAI(max_tokens=200)
                chain = LLMChain(llm=llm, prompt=prompt)
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                    print(cb)
                st.write(response)

                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)
    
if __name__ == '__main__':
    main()
 

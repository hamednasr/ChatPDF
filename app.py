import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatGooglePalm
from streamlit_chat import message
from PyPDF2 import PdfReader

# from google.colab import userdata
# secretName = userdata.get('secretName')

# llm = ChatGooglePalm(google_api_key=secretName, temperature=0.6)
llm = ChatGooglePalm(google_api_key=st.secrets["api_key"], temperature=0.4)

if 'question' not in st.session_state:
  st.session_state.question = None 

if 'chat_history' not in st.session_state:
  st.session_state.chat_history = None 

if 'chain' not in st.session_state:
  st.session_state.chain = None 



styl = f"""
<style>
    .stTextInput {{
      position: fixed;
      bottom: 3rem;
    }}
</style>
"""


def PDF2Text(docs):
  text=''
  for doc in docs:
    reader = PdfReader(doc)
    for page in reader.pages:
      text += page.extract_text()
  return text

def Text2Chunks(text):
  splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n','\n','.',','],
                    chunk_size=700,
                    chunk_overlap=150
                    )

  chunks = splitter.split_text(text)
  return chunks

def Chunks2vectorDB(chunks):
  embeddings = GooglePalmEmbeddings(google_api_key= st.secrets["api_key"])
  vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
  return vectorstore


def CreateChain(vectorDB):
  memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
  chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever = vectorDB.as_retriever(),
    memory=memory
  )
  return chain

def AnswerTheQuestion(question):
  message('wait a second!')
  response = st.session_state.chain({'question': question})
  st.session_state.chat_history = response['chat_history']
  # st.write(st.session_state.chat_history)

  for i,m in enumerate(st.session_state.chat_history):
    if i%2==0:
      message(m.content, is_user=True)
    else:
      message(m.content)

 
def main():
  st.set_page_config(page_title = 'Chat with PDFs', page_icon = ':books:')

  col1, col2 = st.columns(2)

  with col1:
    st.header('Chat with your Documents!')
  with col2:
    st.image('cover.png',width=150)  
    
  st.write('##### First, upload the PDF documents you have and click on Digest! Wait untill processing is finished. Then, you can ask any question about your documents!')
  
  with st.sidebar:
    st.subheader('Upload Documents ')
    docs = st.file_uploader('Upload your PDFs here: ', accept_multiple_files=True)
    button = st.button('Digest!')
  if button:
    with st.spinner('Processing your Docs!'):
      text = PDF2Text(docs)
      chunks = Text2Chunks(text)
      vectorDB = Chunks2vectorDB(chunks)  
      st.session_state.chain = CreateChain(vectorDB)
      st.write(st.session_state.chain.prompt.template)

  question = st.text_input('Question your PDFs here!:  ')
  # placeholder = st.empty()


  st.markdown(styl, unsafe_allow_html=True)

  if question:
    # placeholder.write('##### please wait a second!')
    AnswerTheQuestion(question)
    # placeholder.write('')

if __name__ == '__main__':
  main()

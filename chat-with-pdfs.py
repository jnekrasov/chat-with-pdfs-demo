import platform
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

class bcolors:
    GREEN = '\033[92m'
    ENDCOLOR = '\033[0m'
    
# Load, convert to text and split pdf file into pages
loader = PyPDFLoader("Tesla_Annual_Report_2023_Jan31.pdf")
pages = loader.load_and_split()

# Chunk each page into sections
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len)
sections = text_splitter.split_documents(pages)

# Create FAISS index
faiss_index = FAISS.from_documents(sections, OpenAIEmbeddings())

# Define chain
retriever = faiss_index.as_retriever()
memory = ConversationBufferMemory(
    memory_key='chat_history', 
    return_messages=True, 
    output_key='answer')

chain = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(),
    retriever=retriever,
    memory=memory)


# Collect user input and simulate chat with pdf
if platform.system() == "Windows":
    eof_key = "<Ctrl+Z>"
else:
    eof_key = "<Ctrl+D>"

print(f'Lets talk with Tesla annual report 2023. What would you like to know? Or press {eof_key} to exit.')
while True:
    try:
        user_input = input('Q:')
        print(f"{bcolors.GREEN} A: {chain({'question': user_input})['answer'].strip()}{bcolors.ENDCOLOR}")
    except EOFError:
        break
    except KeyboardInterrupt:
        break

print("Bye!")


#print(chain({'question': 'What was Tesla total revenues and net income?'}))
#print(chain({'question': 'Sum these values?'}))
#print(chain({'question': 'What was the main risk factors for Tesla?'}))
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

PERSIST_DIRECTORY = "db"

class RAGRetriever:

    def __init__(self):
        embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'}
        )
        
        self.vector_store = None

    def _load_documents(self):
        documents_path = os.getenv("DOCUMENTS_PATH")
        
        loader = DirectoryLoader(documents_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        
        print("Carregando documentos...")
        documents = loader.load()
        print(f"{len(documents)} documento(s) carregado(s).")
        return documents

    def _split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200 
        )
        
        print("Dividindo documentos em chunks...")
        chunks = text_splitter.split_documents(documents)
        print(f"Total de {len(chunks)} chunks criados.")
        return chunks

    def setup_vector_store(self, force_recreate=False):
        if os.path.exists(PERSIST_DIRECTORY) and not force_recreate:
            print("Carregando banco de dados vetorial existente...")
            
            self.vector_store = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=self.embeddings
            )
            print("Banco de dados carregado com sucesso.")
        else:
            print("Criando novo banco de dados vetorial...")
            documents = self._load_documents()
            chunks = self._split_documents(documents)
            
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
            print("Banco de dados vetorial criado e salvo com sucesso.")

    def retrieve_context(self, query: str, k: int = 5):
        
        if self.vector_store is None:
            self.setup_vector_store()

        print(f"Recuperando contexto para a query: '{query}'")
        
        retrieved_docs = self.vector_store.similarity_search(query, k=k)
        
        print(f"{len(retrieved_docs)} documentos relevantes recuperados.")
        return retrieved_docs


if __name__ == '__main__':
    
    retriever = RAGRetriever()
    
    
    
    print("Iniciando setup do banco vetorial...")
    retriever.setup_vector_store(force_recreate=True)
    
    print("\nSetup concluído. Realizando uma query de teste...")
    
    
    sample_query = "Qual é o principal objetivo do documento?"
    
    
    relevant_docs = retriever.retrieve_context(sample_query)
    
    
    print("\n--- Resultados da Busca por Similaridade ---")
    for i, doc in enumerate(relevant_docs):
        print(f"\n--- Documento Relevante #{i+1} ---")
        print(f"Fonte: {doc.metadata.get('source', 'N/A')}")
        
        print(f"Conteúdo: {doc.page_content[:250]}...")
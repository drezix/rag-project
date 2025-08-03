import os
import json
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

class RAGRetriever:
    
    def __init__(self, db_path: str = "db"):
        """
        Inicializa o Retriever.
        
        Args:
            db_path (str): O caminho para o diretório do banco de dados vetorial.
                           O padrão é "db" para uso normal.
        """
        self.db_path = db_path  
        embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None 

    def _load_and_chunk_documents(self, chunk_size: int, chunk_overlap: int):
        DATA_DIRECTORY = 'data'
        if not os.path.exists(DATA_DIRECTORY): return []

        print(f"Lendo e dividindo documentos com chunk_size={chunk_size}, chunk_overlap={chunk_overlap}...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        all_documents = []

        for filename in os.listdir(DATA_DIRECTORY):
            if filename.endswith('.json'):
                json_path = os.path.join(DATA_DIRECTORY, filename)
                print(f"  -> Processando arquivo: {filename}")
                
                with open(json_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                global_metadata = {
                    'source_url': data.get('metadata', {}).get('source_url', 'N/A'),
                    'title': data.get('metadata', {}).get('title', 'Sem Título'),
                    'source_file': filename
                }

                for section in data.get('content_sections', []):
                    self._process_section(section, global_metadata, text_splitter, all_documents)

        print(f"Processamento concluído. {len(all_documents)} chunks criados.")
        return all_documents

    def _process_section(self, section_data, global_metadata, text_splitter, all_documents):
        section_title = section_data.get('section_title', 'Sem Título')
        section_text_content = ""
        for item in section_data.get('content', []):
            if item.get('type') == 'paragraph':
                section_text_content += item.get('text', '') + "\n\n"
            elif item.get('type') == 'works_list':
                section_text_content += f"Categoria: {item.get('category', '')}\n"
                for work in item.get('items', []):
                    year = work.get('year')
                    title = work.get('title')
                    section_text_content += f"- {title} ({year})\n" if year else f"- {title}\n"
                section_text_content += "\n"
        
        if section_text_content.strip():
            full_text_with_title = f"Título da Seção: {section_title}\n\n{section_text_content.strip()}"
            chunks = text_splitter.split_text(full_text_with_title)
            for chunk in chunks:
                all_documents.append(Document(
                    page_content=chunk,
                    metadata={**global_metadata, 'section': section_title}
                ))

        for subsection in section_data.get('subsections', []):
            self._process_subsection(subsection, global_metadata, section_title, text_splitter, all_documents)

    def _process_subsection(self, subsection_data, global_metadata, parent_section_title, text_splitter, all_documents):
        subsection_title = subsection_data.get('subsection_title', 'Sem Subtítulo')
        subsection_text_content = ""
        for item in subsection_data.get('content', []):
            if item.get('type') == 'paragraph':
                subsection_text_content += item.get('text', '') + "\n\n"

        if subsection_text_content.strip():
            full_text_with_title = f"Título da Seção: {parent_section_title}\nTítulo da Subseção: {subsection_title}\n\n{subsection_text_content.strip()}"
            chunks = text_splitter.split_text(full_text_with_title)
            for chunk in chunks:
                all_documents.append(Document(
                    page_content=chunk,
                    metadata={**global_metadata, 'section': parent_section_title, 'subsection': subsection_title}
                ))

    def setup_vector_store(self, force_recreate=False, chunk_size=1000, chunk_overlap=200):
        
        if os.path.exists(self.db_path) and not force_recreate:
            print(f"Carregando banco de dados vetorial existente de '{self.db_path}'...")
            self.vector_store = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
            print("Banco de dados carregado com sucesso.")
        else:
            print(f"Criando novo banco de dados vetorial em '{self.db_path}'...")
            documents = self._load_and_chunk_documents(chunk_size, chunk_overlap)
            
            if not documents:
                print("Nenhum documento para indexar. Abortando.")
                return
                
            self.vector_store = Chroma.from_documents(documents=documents, embedding=self.embeddings, persist_directory=self.db_path)
            print("Banco de dados vetorial criado e salvo com sucesso.")

    def retrieve_context(self, query: str, k: int = 5):
        if self.vector_store is None:
            default_size = int(os.getenv("CHUNK_SIZE", 1000))
            default_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
            self.setup_vector_store(chunk_size=default_size, chunk_overlap=default_overlap)
        
        if self.vector_store is None:
            print("Erro: vector_store não foi inicializado.")
            return []

        print(f"Recuperando contexto para a query: '{query}'")
        retrieved_docs = self.vector_store.similarity_search(query, k=k)
        print(f"{len(retrieved_docs)} documentos relevantes recuperados.")
        return retrieved_docs
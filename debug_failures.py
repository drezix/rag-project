import os
import json
import shutil
from app.rag.retriever import RAGRetriever

BEST_CHUNK_SIZE = 250
BEST_CHUNK_OVERLAP = 150
BEST_K = 5

QUESTIONS_FILE = "evaluation_questions.json"
with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
    test_questions = json.load(f)

def debug_failures():
    """
    Roda o retriever com a melhor configuração e analisa em detalhe
    apenas as perguntas que falham.
    """
    print("--- INICIANDO SESSÃO DE DEPURAÇÃO DO RETRIEVER ---")  
    
    db_path = f"db_size_{BEST_CHUNK_SIZE}_overlap_{BEST_CHUNK_OVERLAP}"  
    
    if os.path.exists(db_path):
        shutil.rmtree(db_path)   
    
    print(f"Configurando o retriever com: size={BEST_CHUNK_SIZE}, overlap={BEST_CHUNK_OVERLAP}")
    retriever = RAGRetriever(db_path=db_path)
    retriever.setup_vector_store(
        force_recreate=True, 
        chunk_size=BEST_CHUNK_SIZE, 
        chunk_overlap=BEST_CHUNK_OVERLAP
    )
    
    print("\n--- ANALISANDO PERGUNTAS DE TESTE ---")
    
    total_success = 0

    for i, item in enumerate(test_questions):
        question = item["question"]
        expected_text = item["expected_text"]
        
        retrieved_docs = retriever.retrieve_context(question, k=BEST_K)
        full_context_text = " ".join([doc.page_content for doc in retrieved_docs])
        
        
        if expected_text.lower() in full_context_text.lower():
            print(f"[ SUCESSO ] Pergunta #{i+1}: '{question}'")
            total_success += 1
        else:
            
            print(f"\n\n==================== [ FALHA DETECTADA ] ====================")
            print(f"PERGUNTA #{i+1}: '{question}'")
            print(f"TEXTO ESPERADO (NÃO ENCONTRADO): '{expected_text}'")
            print("\n--- CONTEXTO RECUPERADO (INCORRETO) ---")
            for doc_num, doc in enumerate(retrieved_docs):
                print(f"\n--- Documento Relevante #{doc_num+1} ---")
                print(f"METADADOS: {doc.metadata}")
                print(f"CONTEÚDO:\n{doc.page_content}")
            print("===========================================================\n")

    accuracy = (total_success / len(test_questions)) * 100
    print("\n--- DEPURAÇÃO CONCLUÍDA ---")
    print(f"Precisão final com a melhor configuração: {accuracy:.2f}% ({total_success}/{len(test_questions)})")

if __name__ == "__main__":
    debug_failures()
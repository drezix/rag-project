import os
import json
import shutil
from app.rag.retriever import RAGRetriever

CHUNK_SIZES = [5, 10, 25, 50, 100, 250, 500, 750, 1000]
CHUNK_OVERLAPS = [3, 5, 10, 25, 50, 100, 150, 200, 350, 500]
TOP_K_VALUES = [3, 5, 7, 10]

QUESTIONS_FILE = "evaluation_questions.json"
with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
    test_questions = json.load(f)

def evaluate_retriever():
    """
    Orquestra um experimento completo e robusto, testando chunk_size, 
    chunk_overlap e o número de documentos recuperados (k).
    """
    print("--- INICIANDO AVALIAÇÃO AVANÇADA DO RETRIEVER ---")
    
    results = []

    for size in CHUNK_SIZES:
        for overlap in CHUNK_OVERLAPS:
        
            if overlap >= size:
                continue
        
            print(f"\n--- CRIANDO DB PARA: chunk_size={size}, chunk_overlap={overlap} ---")
            
            db_path_for_run = f"db_size_{size}_overlap_{overlap}"
            if os.path.exists(db_path_for_run):
                shutil.rmtree(db_path_for_run)
            
            retriever = RAGRetriever(db_path=db_path_for_run)
            retriever.setup_vector_store(
                force_recreate=True, 
                chunk_size=size, 
                chunk_overlap=overlap
            )
        
            for k in TOP_K_VALUES:
                print(f"--- AVALIANDO com k={k} ---")
                success_count = 0
                
                for item in test_questions:
                    question = item["question"]
                    expected_text = item["expected_text"]
                    
                    retrieved_docs = retriever.retrieve_context(question, k=k)
                    
                    full_context_text = " ".join([doc.page_content for doc in retrieved_docs])
                    
                    if expected_text.lower() in full_context_text.lower():
                        success_count += 1
                
                accuracy = (success_count / len(test_questions)) * 100
                print(f"Precisão para (size={size}, overlap={overlap}, k={k}): {accuracy:.2f}%")
                
                results.append({
                    "chunk_size": size,
                    "chunk_overlap": overlap,
                    "k": k,
                    "accuracy": accuracy
                })

    print("\n\n--- AVALIAÇÃO CONCLUÍDA ---")
    
    sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    
    print("Melhores configurações encontradas:")
    for res in sorted_results[:10]:
        print(f"  - Tamanho: {res['chunk_size']}, Sobreposição: {res['chunk_overlap']}, K: {res['k']} -> Precisão: {res['accuracy']:.2f}%")
        
    print("\nLembre-se de apagar as pastas 'db_size_*' manualmente quando terminar.")

if __name__ == "__main__":
    evaluate_retriever()
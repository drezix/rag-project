import sys
from app.rag.pipeline import RAGPipeline

def main():
    """
    Função principal para executar o pipeline RAG a partir do terminal.
    """
    # Inicializa o pipeline. Isso irá carregar o retriever e o nosso
    # novo gerador baseado no Gemini.
    rag_pipeline = RAGPipeline()
    
    # Prepara o banco de dados vetorial (carrega se já existir)
    print("\nVerificando o banco de dados vetorial...")
    rag_pipeline.retriever.setup_vector_store()
    print("Banco de dados pronto para uso.")

    print("\n🤖 Assistente de Documentos com Gemini está pronto!")
    print("Digite sua pergunta ou 'sair' para terminar.")

    while True:
        # Pega a pergunta do usuário
        question = input("\nSua pergunta: ")

        if question.lower() == 'sair':
            print("Até logo! 👋")
            break
        
        if not question:
            continue

        print("\n--- DEBUG: VERIFICANDO O QUE O RETRIEVER ESTÁ ENCONTRANDO ---")
        retrieved_docs = rag_pipeline.retriever.retrieve_context(question)
        if retrieved_docs:
            print(f"O retriever encontrou {len(retrieved_docs)} documentos:")
            for i, doc in enumerate(retrieved_docs):
                print(f"\n--- Documento #{i+1} ---")
                print(f"METADADOS: {doc.metadata}")
                print(f"CONTEÚDO (início): {doc.page_content[:350]}...")
        else:
            print("O retriever NÃO encontrou nenhum documento relevante.")
        print("--- FIM DO DEBUG ---\n")

        # Usa o pipeline para obter a resposta
        answer = rag_pipeline.ask(question)

        # Imprime a resposta final
        print("\n--- RESPOSTA ---")
        print(answer)
        print("----------------")

if __name__ == '__main__':
    main()
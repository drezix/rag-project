from .retriever import RAGRetriever
from .generator import RAGGenerator

class RAGPipeline:
    def __init__(self):
        print("Inicializando o pipeline RAG...")
        self.retriever = RAGRetriever()
        self.generator = RAGGenerator()
        print("Pipeline RAG inicializado com sucesso.")

    def _format_context(self, context_docs: list) -> str:
        return "\n\n".join(doc.page_content for doc in context_docs)

    def _create_prompt(self, context: str, question: str) -> str:
        """
        Cria o prompt final para o gerador. 
        Detecta se a pergunta é sobre contagem e usa um template especializado.
        """
        
        prompt_de_contagem = False
        palavras_chave_contagem = ['quantos', 'quantidade', 'número de', 'liste e conte']
        if any(keyword in question.lower() for keyword in palavras_chave_contagem):
            prompt_de_contagem = True

        if prompt_de_contagem:
            print("INFO: Usando prompt especializado para contagem.")
            prompt_template = f"""
Você é um assistente de IA especialista em analisar e contar itens em um texto.

**Tarefa:**
Sua única tarefa é contar o número total de obras literárias listadas no contexto abaixo e fornecer o número exato.

**Instruções Precisas:**
1.  O contexto contém uma lista de obras, separadas por categorias (Romance, Contos, etc.).
2.  Conte cada obra individualmente. Cada linha que começa com um hífen (-) representa uma obra.
3.  **NÃO** conte os nomes das categorias (como "Romance", "Contos") como se fossem obras.
4.  No final, forneça o número total exato. Você pode, opcionalmente, detalhar a contagem por categoria.
5.  Baseie-se **estritamente** na lista fornecida no contexto.

**Contexto:**
---
{context}
---

**Pergunta:**
{question}

**Análise de Contagem:**
"""
        else:
            
            prompt_template = f"""
Você é um assistente de IA especialista em analisar documentos. Sua tarefa é responder à pergunta do usuário usando o contexto fornecido.
Baseie sua resposta apenas nos fatos encontrados nos documentos. Se a informação não estiver presente, informe que não encontrou a resposta nos documentos.

Contexto Fornecido:
---
{context}
---

Pergunta do Usuário:
{question}

Resposta Analítica:
"""
        return prompt_template

    def ask(self, question: str) -> str:
        
        retrieved_context_docs = self.retriever.retrieve_context(question)
        
        formatted_context = self._format_context(retrieved_context_docs)
        
        final_prompt = self._create_prompt(formatted_context, question)

        print("\n\n--- PROMPT COMPLETO ENVIADO PARA O GEMINI ---")
        print(final_prompt)
        print("--- FIM DO PROMPT ---\n\n")

        final_answer = self.generator.generate_response(final_prompt)
        
        return final_answer

if __name__ == '__main__':
    rag_pipeline = RAGPipeline()
    
    print("\n--- INICIANDO TESTE 1 ---")
    question1 = "Quem foi Clarice Lispector?"
    answer1 = rag_pipeline.ask(question1)
    print("\n--- RESPOSTA FINAL 1 ---")
    print(answer1)
    
    print("\n--- INICIANDO TESTE 2 ---")
    question2 = "Qual o nome do primeiro romance de Clarice Lispector?"
    answer2 = rag_pipeline.ask(question2)
    print("\n--- RESPOSTA FINAL 2 ---")
    print(answer2)

    print("\n--- INICIANDO TESTE 3 ---")
    question3 = "Qual era a cor favorita de Clarice Lispector?"
    answer3 = rag_pipeline.ask(question3)
    print("\n--- RESPOSTA FINAL 3 ---")
    print(answer3)
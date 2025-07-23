import os
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

load_dotenv()

class RAGGenerator:
    def __init__(self):
        generator_model_name = os.getenv("GENERATOR_MODEL_NAME")
        
        print(f"Carregando o modelo gerador: {generator_model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device) 
        
        print(f"Modelo carregado e rodando em: {self.device.upper()}")

    def generate_response(self, prompt: str) -> str:
        
        print("Gerando resposta...")
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs.to(self.device) 
        
        output_tokens = self.model.generate(
            inputs["input_ids"], 
            max_length=150, 
            num_beams=5, 
            early_stopping=True
        )
        
        response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        print("Resposta gerada com sucesso.")
        return response

if __name__ == '__main__':
    
    generator = RAGGenerator()
    
    sample_context = """
    Clarice Lispector (1920-1977) foi uma escritora e jornalista brasileira, nascida na Ucrânia.
    É considerada uma das mais importantes escritoras do século XX. Sua obra é caracterizada
    pela introspecção psicológica e por um estilo de escrita inovador, com foco no fluxo de
    consciência de seus personagens. Seu primeiro romance, "Perto do Coração Selvagem",
    publicado em 1943, recebeu o Prêmio Graça Aranha.
    """
    
    
    sample_question = "Qual o nome da escritora de Perto do Coração Selvagem?"
    
    
    test_prompt = f"""Extraia a resposta para a pergunta a partir do contexto.

    Contexto: {sample_context}

    Pergunta: {sample_question}
    """
    
    
    print("\n--- Enviando prompt de teste para o gerador ---")
    print(f"Prompt:\n{test_prompt}")
    
    generated_answer = generator.generate_response(test_prompt)
    
    print("\n--- Resposta Gerada ---")
    print(generated_answer)
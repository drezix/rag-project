import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

class RAGGenerator:
    def __init__(self):
        """
        Configura o cliente da API do Gemini.
        """
        print("Configurando o gerador com a API do Gemini...")
        
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("API key do Google não encontrada. Verifique seu arquivo .env")
            
            genai.configure(api_key=api_key)
            
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("Gerador Gemini configurado com sucesso. ✨")

        except Exception as e:
            print(f"Erro ao configurar o gerador Gemini: {e}")
            self.model = None

    def generate_response(self, prompt: str) -> str:
        """
        Gera uma resposta usando o modelo Gemini a partir de um prompt.
        """
        if not self.model:
            return "Erro: O modelo gerador não foi inicializado corretamente."
            
        print("Gerando resposta com o Gemini...")
        
        try:
            response = self.model.generate_content(prompt)
            
            if not response.parts:
                 return "A resposta foi bloqueada devido às políticas de segurança. Tente reformular a pergunta."
            
            generated_text = response.text
            
            print("Resposta gerada com sucesso.")
            return generated_text
        except Exception as e:
            print(f"Erro ao chamar a API do Gemini: {e}")
            return f"Desculpe, ocorreu um erro ao gerar a resposta: {e}"
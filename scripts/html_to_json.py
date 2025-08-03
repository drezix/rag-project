import requests
from bs4 import BeautifulSoup
import json
import re
import os

def parse_infobox(soup):
  """Extrai os dados da tabela de infobox da Wikipédia de forma mais robusta."""
  infobox = soup.find('table', class_='infobox')
  if not infobox:
    return {}

  data = {}
  for row in infobox.find_all('tr'):
    header = row.find('th')
    value_cell = row.find('td')

    if header and value_cell:
      key = header.get_text(strip=True)
  
      value = value_cell.get_text(separator=' ', strip=True)
  
      value_cleaned = re.sub(r'\s+', ' ', re.sub(r'\[.*?\]', '', value)).strip()
      
      if key and value_cleaned:
        data[key] = value_cleaned
  return data

def scrape_wikipedia_to_structured_json(url: str, output_filename: str = 'dados_clarice_final.json'):
  """
  Acessa uma URL, extrai o infobox e o conteúdo principal de forma hierárquica 
  e salva em um único arquivo JSON bem estruturado.
  """
  print(f"1. Acessando a URL: {url}")
  try:
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
  except requests.exceptions.RequestException as e:
    print(f"Erro ao acessar a URL: {e}")
    return

  print("2. HTML baixado. Iniciando análise estruturada completa...")
  soup = BeautifulSoup(response.content, 'html.parser')
  
  final_data = {
    "metadata": {
      "source_url": url,
      "title": soup.find('h1', id='firstHeading').get_text(strip=True),
      "infobox": parse_infobox(soup)
    },
    "content_sections": []
  }

  content_div = soup.find(id="mw-content-text").find('div', class_='mw-parser-output')
  if not content_div:
    return

  current_h2_section = None
  current_h3_section = None
  
  tags_de_interesse = ['h2', 'h3', 'p', 'ul']
  
  for element in content_div.find_all(tags_de_interesse):
    
    if element.find_parent('table', class_='infobox'):
      continue

    if element.name == 'h2':
      section_title = element.get_text(strip=True).replace('[editar | editar código-fonte]', '')
      if section_title in ["Ver também", "Notas", "Referências", "Ligações externas", "Bibliografia"]:
        break
      
      current_h2_section = {"section_title": section_title, "content": []}
      final_data["content_sections"].append(current_h2_section)
      current_h3_section = None
    
    elif element.name == 'h3' and current_h2_section:
      subsection_title = element.get_text(strip=True).replace('[editar | editar código-fonte]', '')
      current_h3_section = {"subsection_title": subsection_title, "content": []}
      
      if "subsections" not in current_h2_section:
        current_h2_section["subsections"] = []
      current_h2_section["subsections"].append(current_h3_section)
    
    elif element.name == 'p' and current_h2_section:
      text = re.sub(r'\s+', ' ', re.sub(r'\[.*?\]', '', element.get_text(strip=True))).strip()
      if text:
        content_to_add = {"type": "paragraph", "text": text}
        if current_h3_section:
          current_h3_section["content"].append(content_to_add)
        else:
          current_h2_section["content"].append(content_to_add)

    elif element.name == 'ul' and current_h2_section:
    
      if current_h2_section["section_title"] == "Lista de obras":
        category_tag = element.find_previous_sibling(['h2', 'h3', 'p'])
        category_name = category_tag.get_text(strip=True) if category_tag else "Sem Categoria"
        
        works = []
        for li in element.find_all('li'):
          match = re.search(r'^(.*?)\s*\((\d{4})\)', li.get_text(strip=True))
          if match:
            works.append({"title": match.group(1).strip(), "year": match.group(2)})
          else:
            works.append({"title": li.get_text(strip=True), "year": None})
        
    
        current_h2_section["content"].append({
          "type": "works_list",
          "category": category_name,
          "items": works
        })

  print(f"3. Análise concluída. {len(final_data['content_sections'])} seções principais encontradas.")
  
  data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
  os.makedirs(data_dir, exist_ok=True)
  output_path = os.path.join(data_dir, output_filename)
  
  with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)
    
  print(f"✅ Sucesso! Os dados completos foram salvos no arquivo '{output_path}'")

if __name__ == '__main__':
  wikipedia_url = "https://pt.wikipedia.org/wiki/Clarice_Lispector"
  scrape_wikipedia_to_structured_json(wikipedia_url)
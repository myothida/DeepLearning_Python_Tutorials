import requests
from bs4 import BeautifulSoup

def extract_medium_post_content(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()  
        soup = BeautifulSoup(response.text, 'html.parser') 
        title = soup.find('h1').get_text(strip=True) if soup.find('h1') else "No title found"
        article = soup.find('article') 
        if not article:
            return {"error": "Could not find the main article content"}

        content_parts = article.find_all(['p'])
        content = "\n".join([part.get_text(strip=True) for part in content_parts])
        index = content.find('Share')
        content = content[index+6:]
        return content
            
    except requests.exceptions.RequestException as e:
        return {"error": f"Request error: {str(e)}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}




from typing import List, Dict, Literal, Optional
import google.generai as genai
import openai
import requests
import re

class OnLineLLMs:
    def __init__(
        self,
        model_name: Literal["gemini", "openai", "together", "chatgroq"],
        api_key: str,
        model_version: str,
        base_url: Optional[str] = None
    ):
        """Initialize model with the specified name, API key, and model version."""
        self.model_name = model_name.lower()
        self.model_version = model_version

        if self.model_name == "gemini" and api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name=model_version)

        elif self.model_name == "openai" and api_key:
            self.client = openai.OpenAI(api_key=api_key)

        elif self.model_name == "together" and api_key:
            if not base_url:
                raise ValueError("Together API requires base_url, e.g. https://api.together.xyz")
            self.base_url = f"{base_url}/v1/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

        elif self.model_name == "chatgroq" and api_key:
            self.base_url = f"{base_url}/openai/v1/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

        else:
            raise ValueError("Unsupported model name or missing API key.")

    def remove_think_blocks(self, text: str) -> str:
        """Remove <think> blocks and their content from text."""
        pattern = r'<think>.*?</think>'
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
        cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text).strip()
        return cleaned_text

    def generate_content(self, prompt: List[Dict[str, str]]) -> str:
        """Generate content using the online LLM based on the provided prompt."""
        if self.model_name == "gemini":
            gemini_messages = [
                {"role": msg["role"], "parts": [msg["content"]]} for msg in prompt
            ]
            response = self.model.generate_content(gemini_messages)
            try:
                return response.text
            except AttributeError:
                return response.candidates[0].content.parts[0].text

        elif self.model_name == "openai":
            response = self.client.chat.completions.create(
                model=self.model_version,
                messages=prompt
            )
            return response.choices[0].message.content

        elif self.model_name == "together":
            data = {
                "model": self.model_version,
                "messages": prompt,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 512,
            }
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            response_data = response.json()["choices"][0]["message"]["content"].strip()
            return self.remove_think_blocks(response_data)

        elif self.model_name == "chatgroq":
            data = {
                "model": self.model_version,
                "messages": prompt,
                "temperature": 0.1,
                "top_p": 0.9,
            }
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            response_data = response.json()["choices"][0]["message"]["content"].strip()
            return self.remove_think_blocks(response_data)

        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        

if __name__ == '__main__':
    llm = OnLineLLMs(
    model_name="chatgroq",
    api_key="YOUR-API-KEY",
    model_version="llama-3.1-8b-instant",
    base_url="https://api.groq.com"
)

    response = llm.generate_content([
        {"role": "user", "content": "Explain the difference between CPU and GPU."}
    ])
    print(response)

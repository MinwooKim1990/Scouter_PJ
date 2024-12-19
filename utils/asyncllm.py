import asyncio
import aiohttp
from typing import Any, Dict
from google import genai

class LLMClient:
    def __init__(self, api_key: str, provider: str):
        self.api_key = api_key
        self.provider = provider.lower()

    async def call_model(self, prompt: str, model: str) -> str:
        if self.provider == "groq":
            return await self._call_groq(prompt, model)
        elif self.provider == "openai":
            return await self._call_openai(prompt, model)
        elif self.provider == "google":
            return self._call_google(prompt, model)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _call_groq(self, prompt: str, model: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Error {response.status}: {await response.text()}")
                data = await response.json()
                return data['choices'][0]['message']['content']

    async def _call_openai(self, prompt: str, model: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 100
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Error {response.status}: {await response.text()}")
                data = await response.json()
                return data['choices'][0]['message']['content']

    def _call_google(self, prompt: str, model: str) -> str:
        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=model, contents=prompt
        )
        return response.text

# Example usage
async def main():
    user_input = "Your response will show in small screen. Shorten your response to 3 or 4 sentences about this user prompt:[Explain the importance of fast language models]"
    api_key = "api key"
    provider = "openai"  # Options: "groq", "openai", "google"
    model = "gpt-4o-mini-2024-07-18"  # Replace with actual model name for the provider
    #model = "llama3-8b-8192"  
    #model = 'gemini-2.0-flash-exp'

    client = LLMClient(api_key=api_key, provider=provider)
    try:
        response = await client.call_model(prompt=user_input, model=model)
        print("Response:", response)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

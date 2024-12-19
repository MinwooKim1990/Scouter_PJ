import asyncio
import aiohttp
import anthropic
from typing import Any, Dict, List, Optional
from google import genai

class Conversation:
    def __init__(self, max_tokens: int = 4000):
        self.messages: List[Dict[str, str]] = []
        self.max_tokens = max_tokens
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.messages.append({"role": role, "content": content})
        # Basic token estimation (can be improved with proper tokenizer)
        while self._estimate_tokens() > self.max_tokens and len(self.messages) > 2:
            # Keep at least the last user message and response
            self.messages.pop(0)
    
    def _estimate_tokens(self) -> int:
        """Rough estimation of tokens (4 chars ≈ 1 token)"""
        return sum(len(msg["content"]) // 4 for msg in self.messages)
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in the conversation"""
        return self.messages
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []

class LLMClient:
    def __init__(self, api_key: str, provider: str):
        self.api_key = api_key
        self.provider = provider.lower()
        self.conversation = Conversation()
        if self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=api_key)

    async def call_model(self, prompt: str, model: str) -> str:
        """Call the appropriate LLM based on provider"""
        # Add user message to conversation
        self.conversation.add_message("user", prompt)
        
        try:
            if self.provider == "groq":
                response = await self._call_groq(prompt, model)
            elif self.provider == "openai":
                response = await self._call_openai(prompt, model)
            elif self.provider == "google":
                response = self._call_google(prompt, model)
            elif self.provider == "anthropic":
                response = await self._call_anthropic(prompt, model)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Add assistant response to conversation
            self.conversation.add_message("assistant", response)
            return response
        except Exception as e:
            error_msg = f"Error calling {self.provider}: {str(e)}"
            self.conversation.add_message("system", error_msg)
            raise

    async def _call_groq(self, prompt: str, model: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        messages = self._prepare_messages(prompt)
        payload = {
            "model": model,
            "messages": messages
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
        messages = self._prepare_messages(prompt)
        payload = {
            "model": model,
            "messages": messages,
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
        messages = self._prepare_messages(prompt)
        # Convert message history to appropriate format for Google
        prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        response = client.models.generate_content(
            model=model, contents=prompt_text
        )
        return response.text

    async def _call_anthropic(self, prompt: str, model: str) -> str:
        messages = self._prepare_messages(prompt)
        # Convert message history to Anthropic format
        system_prompt = "You are a helpful AI assistant. Your responses will show in a small screen, so keep them concise (3-4 sentences)."
        message_text = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        try:
            completion = await self.client.messages.create(
                model=model,  # e.g., "claude-3-opus-20240229"
                max_tokens=1024,
                messages=[{"role": "user", "content": message_text}],
                system=system_prompt
            )
            return completion.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")

    def _prepare_messages(self, current_prompt: str) -> List[Dict[str, str]]:
        """Prepare messages including conversation history"""
        messages = self.conversation.get_messages()
        if not messages or messages[-1]["content"] != current_prompt:
            messages.append({"role": "user", "content": current_prompt})
        return messages

    def clear_memory(self):
        """Clear conversation history"""
        self.conversation.clear()

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the full conversation history"""
        return self.conversation.get_messages()

# Example usage
async def main():
    prompt = "Your response will show in small screen. Shorten your response to 3 or 4 sentences about this user prompt:[What is quantum computing?]"
    api_key = "your-api-key"
    provider = "anthropic"
    model = "claude-3-opus-20240229"

    client = LLMClient(api_key=api_key, provider=provider)
    try:
        response = await client.call_model(prompt=prompt, model=model)
        print("Response:", response)
        
        # Show conversation history
        print("\nConversation history:")
        for msg in client.get_conversation_history():
            print(f"{msg['role']}: {msg['content']}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

text = """
A stack is a linear data structure that follows LIFO.
Push adds an element to the stack and pop removes the top element.
"""

token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("Add your Hugging Face token as HF_TOKEN in the .env file.")

model = os.getenv("HF_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
client = InferenceClient(api_key=token)

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": f"Create 5 flashcards from this text:\n\n{text}",
        }
    ],
    max_tokens=500,
)
print(response.choices[0].message.content)

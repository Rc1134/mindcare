from dotenv import load_dotenv
import os
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

from huggingface_hub import InferenceClient

client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        token=hf_token
)

response = client.text_generation(
    prompt="What is the capital of India?",
    max_new_tokens=50
)

print(response)

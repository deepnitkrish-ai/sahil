import os
import base64
from groq import Groq

# Load API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

image_path = "acne.jpg"
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

query = "Is there something wrong with my face?"
model = "llama-3.1-8b-instant"

# Groq chat expects messages as a list of dicts, with "content" as a string
messages = [
    {
        "role": "user",
        "content": query  # content must be a string, not a list or dict
    }
]

# Note: Groq models currently do NOT support image inputs,
# so including the image in the messages will cause errors.
# You can still keep the encoded image separately if you want.

chat_response = client.chat.completions.create(
    model=model,
    messages=messages
)

print(chat_response.choices[0].message.content)

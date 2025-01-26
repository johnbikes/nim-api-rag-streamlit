import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

def main():

    load_dotenv()

    nim_api_key = os.environ.get('NIM_API_KEY')
    print(f"nim_api_key = {nim_api_key}")

    client = OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = nim_api_key
    )

    completion = client.chat.completions.create(
        model="meta/llama-3.2-3b-instruct",
        messages=[{"role":"user",
                   "content":"Provide me an article on Machine Learning"
                   }],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        stream=True
    )

    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")

if __name__ == "__main__":
    main()


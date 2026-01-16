import os
import sys
import base64
from dotenv import load_dotenv

from groq import Groq

load_dotenv()


def b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    if len(sys.argv) < 2:
        print("Usage: python 11A_vision_cli_self_contained.py <image_path> [question]")
        sys.exit(1)

    image_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) >= 3 else "Describe this image in detail."

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError('Missing GROQ_API_KEY. Put it in .env or export it.')

    client = Groq(api_key=api_key)

    img_b64 = b64_image(image_path)
    data_url = f"data:image/jpeg;base64,{img_b64}"

    # NOTE: Vision model name must be one your Groq account supports.
    # If this errors, change model to the one you use in your project logs.
    model_name = "meta-llama/llama-4-maverick-17b-128e-instruct"
    model = os.getenv("GROQ_VISION_MODEL", model_name)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful vision assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0.2,
        max_tokens=500,
    )

    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()

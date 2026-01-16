import os
import base64
from dotenv import load_dotenv

import streamlit as st
from PIL import Image
from groq import Groq

load_dotenv()


def b64_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


st.title("Vision Demo (Self-Contained)")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error('Missing GROQ_API_KEY. Put it in a .env file or export it.')
    st.stop()

client = Groq(api_key=api_key)

# Change if needed (or set GROQ_VISION_MODEL in .env)
model_name = "meta-llama/llama-4-maverick-17b-128e-instruct"
model = os.getenv("GROQ_VISION_MODEL", model_name)

img_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
question = st.text_input("Question", "Describe the image and list 5 important details.")

if img_file:
    img = Image.open(img_file)
    st.image(img, use_container_width=True)

    # Make a data URL from the uploaded image bytes
    mime = img_file.type or "image/jpeg"
    data_url = f"data:{mime};base64,{b64_bytes(img_file.getvalue())}"

    if st.button("Ask", type="primary"):
        with st.spinner("Thinking..."):
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
        st.subheader("Answer")
        st.write(resp.choices[0].message.content)
else:
    st.info("Upload an image to begin.")

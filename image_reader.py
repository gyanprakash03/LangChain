from dotenv import load_dotenv
from PIL import Image
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import base64

load_dotenv()

model = ChatGroq(
    model="qwen/qwen3.6-27b",
    reasoning_format="hidden",
    reasoning_effort="none"
)

with open("sxf_page.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": """
                You are an expert manga and scene understanding assistant. Analyze this manga page.

                Return your answer using the following sections.

                ## Characters
                List every visible character.

                ## Dialogue
                Transcribe every speech bubble exactly as written.

                ## Thoughts
                Transcribe every thought bubble exactly as written.

                ## Sound Effects
                List every visible sound effect exactly as written.

                ## Panel-by-panel Description
                Describe what happens in each panel from top-left to bottom-right.

                ## Overall Summary
            """
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_data}"
            }
        }
    ]
)

response = model.invoke([message])

print(response.content)
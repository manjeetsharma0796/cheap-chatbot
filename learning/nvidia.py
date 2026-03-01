## Core LC Chat Interface
from langchain_nvidia_ai_endpoints import ChatNVIDIA

api = "***REMOVED_NVIDIA_API_KEY***"

llm = ChatNVIDIA(model="meta/llama-3.2-90b-vision-instruct", 
                   temperature=0.7, 
                #    max_tokens=2048,
                   api_key=api,
                   )

# result = llm.invoke("Write a ballad about LangChain." )
# print(result.content)

# llm.stream_invoke("Write a ballad about LangChain.", stream=True)

# Streaming response ===========

# ai_res = llm.stream("Write a ballad about LangChain.")

# full_response = ""

# for chunk in ai_res:
#     print(chunk.content, end="",flush=True)
#     full_response += chunk.content
    
    ## Pipelines ==============
    
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# prompt = ChatPromptTemplate.from_messages(
#     [("system", 
#       "You are an expert coding AI. Respond only in valid python; no narration whatsoever."),
#      ("user", 
#       "{input}")]
# )
# chain = prompt | llm | StrOutputParser()

# for txt in chain.stream({"input": "how do i solve fizz buzz problem?"}):
#     print(txt, end="")

import IPython
import requests

image_url = "https://www.nvidia.com/content/dam/en-zz/Solutions/research/ai-playground/nvidia-picasso-3c33-p@2x.jpg"  ## Large Image
image_content = requests.get(image_url).content

# IPython.display.Image(image_content)

from langchain.messages import HumanMessage

ai_res1 = llm.stream(
    [
        HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image:"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )
    ]
)

for chunk in ai_res1:
    print(chunk.content, end="", flush=True)
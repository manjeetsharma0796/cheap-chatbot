import os
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()
api = os.getenv("NVIDIA_API_KEY")

llm = ChatNVIDIA(model="z-ai/glm4.7", 
                   api_key=api,
                   temperature=1,
                   top_p=1,
                   max_tokens=16384,
                #    extra_body={"chat_template_kwargs":{"enable_thinking":True,"clear_thinking":False}},
                   )

# store is a dictionary that maps session IDs to their corresponding chat histories.
store = {}  # memory is maintained outside the chain


# A function that returns the chat history for a given session ID.
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]




#  Define a RunnableConfig object, with a `configurable` key. session_id determines thread
config = {"configurable": {"session_id": "1"}}

conversation = RunnableWithMessageHistory(
    llm,
    get_session_history,
    
)

ai_res =conversation.stream(
    "Hi I'm pandit ji.",  # input or query
    config=config,
)

for chunk in ai_res:
    print(chunk.content, end="", flush=True)

from langchain.tools import tool
from pydantic import Field


@tool
def get_current_weather(
    location: str = Field(..., description="The location to get the weather for."),
):
    """Get the current weather for a location."""
    ...


# llm = llm.bind_tools(tools=[get_current_weather])
response = llm.stream("What is  Boston?")
# response.tool_calls
full =""
for chunk in response:
    print(chunk.content, end="", flush=True)
    full += chunk.content
    
print(full)

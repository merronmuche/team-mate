import uvicorn
import socketio
from fastapi import FastAPI
from typing import Dict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from weaviate import setup_weaviate_interface
import asyncio

# Load environment variables
load_dotenv()

# Instantiate the ChatGPT model
chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

# FastAPI application
app = FastAPI()

# Socket.IO server
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")

# Wrap Socket.IO server with ASGI application
socket_app = socketio.ASGIApp(sio)
app.mount("/", socket_app)

# Dictionary to store session data
sessions: Dict[str, List[Dict[str, str]]] = {}

# Weaviate client placeholder
client = None

# Function to setup Weaviate interface and get the client
async def get_weaviate_client():
    weaviate_interface = await setup_weaviate_interface()
    return weaviate_interface.client

# Endpoint to test server
@app.on_event("startup")
async def on_startup():
    global client
    client = await get_weaviate_client()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Socket.IO event handlers
@sio.on("connect")
async def connect(sid, env):
    print("New Client Connected to This id: " + str(sid))

@sio.on("disconnect")
async def disconnect(sid):
    print("Client Disconnected: " + str(sid))

@sio.on("connectionInit")
async def handle_connection_init(sid):
    await sio.emit("connectionAck", room=sid)

@sio.on("sessionInit")
async def handle_session_init(sid, data):
    print(f"===> Session {sid} initialized")
    session_id = data.get("sessionId")
    if session_id not in sessions:
        sessions[session_id] = []
    print(f"**** Session {session_id} initialized for {sid} session data: {sessions[session_id]}")
    await sio.emit("sessionInit", {"sessionId": session_id, "chatHistory": sessions[session_id]}, room=sid)

# Weaviate search function
async def weaviate_search(query: str, class_name: str = "DocumentChunk") -> List[Dict[str, str]]:
    graphql_query = f"""
    {{
        Get {{
            {class_name}(where: {{
                operator: "Equal",
                path: ["text"],
                valueString: "{query}"
            }}) {{
                text
            }}
        }}
    }}
    """
    search_results = await client.run_query(graphql_query)
    return search_results["data"]["Get"][class_name] if "data" in search_results and "Get" in search_results["data"] else []

# Handle incoming chat messages
@sio.on("textMessage")
async def handle_chat_message(sid, data):
    print(f"Message from {sid}: {data}")
    user_message = data['message']

    # Retrieve relevant documents from Weaviate
    retrieved_docs = await weaviate_search(user_message)
    context = "\n".join([doc["text"] for doc in retrieved_docs])

    # Combine user query with retrieved documents
    augmented_query = f"User query: {user_message}\n\nContext:\n{context}"

    # Implement the GPT-3.5 Turbo API call to generate a response
    response = chat.invoke(
            [
                HumanMessage(content=augmented_query),
            ]
        ).content

    session_id = data.get("sessionId")
    if session_id:
        if session_id not in sessions:
            raise Exception(f"Session {session_id} not found")
        received_message = {
            "id": data.get("id"),
            "message": data.get("message"),
            "isUserMessage": True,
            "timestamp": data.get("timestamp"),
        }
        sessions[session_id].append(received_message)
        response_message = {
            "id": data.get("id") + "_response",
            "textResponse": response,
            "isUserMessage": False,
            "timestamp": data.get("timestamp"),
            "isComplete": True,
        }
        await sio.emit("textResponse", response_message, room=sid)
        sessions[session_id].append(response_message)

        print(f"Message from {sid} in session {session_id}: {data.get('message')}")

    else:
        print(f"No session ID provided by {sid}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=6789, lifespan="on", reload=True)

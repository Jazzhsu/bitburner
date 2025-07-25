import fastapi
import uvicorn
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = fastapi.FastAPI();


origins = [
    # Allow requests from React development server
    "http://localhost:666",
    "http://127.0.0.1:666",
    "https://bitburner-official.github.io",  # <-- Add this line
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/hello_world")
async def hello_world():
    print("hello_world")

if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0.0.0.0", port=666, reload=True)

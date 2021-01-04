import websockets
import asyncio
import json

async def chat(websocket, path):
    await websocket.send(json.dumps({"type": "handshake"}))
    async for message in websocket:
        print(message)


start_server = websockets.serve(chat, "127.0.0.1", 1234)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
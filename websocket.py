# -*- coding: utf-8 -*-
import json
import asyncio
import requests
import websockets

from const import *


async def chat(websocket, path):
    await websocket.send(json.dumps({'type': 'handshake'}))
    async for message in websocket:
        print(message)
        requests.get(f'http://localhost:{FLASK_PORT}/data?coor={message}')


if __name__ == '__main__':
    start_server = websockets.serve(chat, '127.0.0.1', WS_PORT)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

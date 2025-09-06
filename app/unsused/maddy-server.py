import json
import asyncio
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import base64
import os
import websockets
from datetime import datetime
import uuid

app = FastAPI()
print("app created")

active_connections: Dict[str, WebSocket] = {}

async def delayed_welcome(websocket: WebSocket, bot_id: str):
            print("delayed_welcome")
            await asyncio.sleep(15)  # yield control, non-blocking
            try:
                message = {
                                "command": "sendmsg",
                                'message': f'Hello Dropping message from server',
                                'bot_id': bot_id
                            }
                print("sending welcome")
                await websocket.send_text(json.dumps(message))
        
                print(f"[Welcome sent] to {bot_id}")
            except Exception as e:
                print(f"[Welcome error] {bot_id}: {e}")

async def delayed_send_audio(websocket: WebSocket, bot_id: str, audio_folder_path: str):
            print("delayed_send_audio")
            await asyncio.sleep(30)  # yield control, non-blocking
            try:
                # get all the files in the folder
                files = os.listdir(audio_folder_path)
                for file in files:
                    with open(os.path.join(audio_folder_path, file), 'rb') as pcm_file:
                        while True:
                                chunk = pcm_file.read(64000)
                                if not chunk:  # End of file
                                    break
                                base64_audio = base64.b64encode(chunk).decode('utf-8')
                                await websocket.send_text(json.dumps({
                                    "command": "sendaudio",
                                    "audiochunk": base64_audio
                                }))
                                print(f"[Audio chunk sent] {len(base64_audio)}")
                                print("sending audio")
                        print(f"[Audio sent] to {bot_id}")
            except Exception as e:
                print(f"[Audio error] {bot_id}: {e}")


@app.websocket("/twoWayCommunication")
async def websocket_endpoint(websocket: WebSocket):
    
    await websocket.accept()
    bot_id = 'Default-id'
    print("connection accepted")
    
    try:
        init_message = await websocket.receive_json()
        bot_id = init_message.get("bot_id")
        if not bot_id:
            print("bot_id not found")
            await websocket.close(code=1003)
            return

        active_connections[bot_id] = websocket
        print(f"[Connected] {bot_id}")

        # Schedule welcome message after 1 minute
        asyncio.create_task(delayed_welcome(websocket, bot_id))
        # pcm chunks should be in 32000 bitrate
        asyncio.create_task(delayed_send_audio(websocket, bot_id, "chunks"))

        # Main receive loop
        while True:
            print("receiving message")
            message = await websocket.receive_text()
            print("message received")
            try:
                print(message)
                print("message is json")
                data = json.loads(message)
                target_name = data.get("bot_id")
                if target_name and target_name != bot_id:
                    target_ws = active_connections.get(target_name)
                    if target_ws:
                        await target_ws.send_text(message)
                        print(f"[Forwarded] to {target_name}")
                    else:
                        print(f"Target client '{target_name}' not found.")
                else:
                    print("Invalid message format or self-targeting detected.")
            except json.JSONDecodeError:
                print("Received message is not valid JSON.")
            except Exception as e:
                print(f"Error processing message: {e}")

    except WebSocketDisconnect:
        print(f"[Disconnected] {bot_id}")
    except Exception as e:
        print(f"[Error] {e}")

    finally:
        if bot_id and bot_id in active_connections:
            del active_connections[bot_id]
            print(f"[Cleaned up] {bot_id}")
      

@app.websocket("/liveAudioChunks")
async def liveAudioChunks(websocket: WebSocket):
    
    await websocket.accept()
    print("connection accepted")
    
    session_id = str(uuid.uuid4())[:8]
    chunk_counter = 0
    chunks_dir = "chunks"
    os.makedirs(chunks_dir, exist_ok=True)
    
    try:
        init_message = await websocket.receive_json()

        # Main receive loop
        while True:
            print("receiving message")
            message = await websocket.receive_text()

            try:
                data = json.loads(message)
                audio_data = data.get("audioData")
                message_type = data.get("type")
                # add null check
                speakerId = data.get("speakerId") if data.get("speakerId") else "Default-id"
                speakerName = data.get("speakerName") if data.get("speakerName") else "Default-name"
                # timestamp = data.get("timestamp")
                # print("--------------------------------")
                print("message_type", message_type,
                      "audio_data", len(audio_data),
                      "speakerId", speakerId,
                      "speakerName", speakerName)
                if audio_data:
                    # Decode base64 audio data
                    try:
                       
                        pcm_data = base64.b64decode(audio_data)
                        
                        # Generate filename with timestamp and session info
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"audio_{session_id}_{timestamp}_{chunk_counter:04d}.pcm"
                        filepath = os.path.join(chunks_dir, filename)
                        
                        # Save PCM data to file
                        with open(filepath, 'wb') as pcm_file:
                            pcm_file.write(pcm_data)
                        
                        chunk_counter += 1
                        print(f"Saved audio chunk: {filename} ({len(pcm_data)} bytes)")
                    
                    except Exception as decode_error:
                        print(f"Error decoding/saving audio data: {decode_error}")
                
                print("received message type", message_type, "audio data length:", len(audio_data) if audio_data else 0)
                
            except Exception as e:
                print(f"Error processing message: {e}")

    except WebSocketDisconnect:
        print(f"[Disconnected]")
    except Exception as e:
        print(f"[Error] {e}")
  
            
# IN ubuntu 24.04
# source venv
# uvicorn server:app --port 9000
# pip3 install fastapi uvicorn
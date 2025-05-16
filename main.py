import fastapi
import uvicorn
import websockets
import httpx
import openai
import os
import io
import tempfile
import json
import logging
import asyncio
import base64
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketState
from elevenlabs.client import ElevenLabs
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create static directory if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in .env file")
    raise ValueError("OPENAI_API_KEY not found in .env file")
if not ELEVENLABS_API_KEY:
    logger.error("ELEVENLABS_API_KEY not found in .env file")
    raise ValueError("ELEVENLABS_API_KEY not found in .env file")

openai.api_key = OPENAI_API_KEY
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

app = FastAPI()

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Helper Functions
async def transcribe_with_whisper(audio_bytes: bytes) -> str:
    """Transcribes audio using OpenAI Whisper API."""
    if not audio_bytes:
        logger.warning("Transcribe_with_whisper called with empty audio_bytes.")
        return ""
    logger.info(f"Attempting Whisper transcription for audio of size: {len(audio_bytes)} bytes.")
    try:
        async_openai_client = openai.AsyncOpenAI()
        # Using a temporary file is still a good approach for handling raw bytes with the API
        with tempfile.NamedTemporaryFile(suffix=".webm", mode='wb', delete=False) as tmp_audio_file:
            tmp_audio_file.write(audio_bytes)
            tmp_audio_file_name = tmp_audio_file.name # Get the file name
        
        # The OpenAI v1.x library expects a PathLike object (like the filename string) 
        # or an io.IOBase object opened in binary read mode, or bytes.
        # Here, we pass the filename. The library will handle opening/reading.
        try:
            with open(tmp_audio_file_name, "rb") as file_to_transcribe:
                 transcript = await async_openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=file_to_transcribe, # Pass the opened file object
                )
            logger.info(f"Whisper transcription successful. Transcript: {transcript.text[:50]}...")
            return transcript.text
        finally:
            # Ensure the temporary file is deleted
            if os.path.exists(tmp_audio_file_name):
                os.remove(tmp_audio_file_name)

    except Exception as e:
        logger.error(f"Error in Whisper transcription: {e}", exc_info=True)
        # Clean up the temp file in case of an error before the finally block
        if 'tmp_audio_file_name' in locals() and os.path.exists(tmp_audio_file_name):
            try:
                os.remove(tmp_audio_file_name)
            except Exception as cleanup_e:
                logger.error(f"Error cleaning up temp file {tmp_audio_file_name}: {cleanup_e}")
        return ""

async def chat_with_gpt(text: str, conversation_history: list = None) -> str:
    """Gets a chatbot reply from OpenAI Chat API."""
    if not text:
        logger.warning("Chat_with_gpt called with empty text.")
        return "I didn't catch that. Could you please repeat?"
    
    logger.info(f"Sending text to GPT with conversation history. Text: {text[:100]}...")
    
    # Base system message
    system_message = {
        "role": "system", 
        "content": """You are a professional hotel receptionist at a luxury 5-star hotel. Follow these guidelines:
- Greet guests warmly but efficiently
- Handle check-ins, check-outs, and room inquiries
- Provide information about hotel amenities (pool, spa, restaurant, room service)
- Address guest concerns promptly and professionally
- Keep responses under 30 words and be direct
- Maintain a polite, helpful tone
- If asked about room rates: Standard ($200), Deluxe ($350), Suite ($500)
- Breakfast: 6:30-10:30 AM, Restaurant: 12-11 PM, Pool: 7 AM-9 PM
- IMPORTANT: Maintain context from the conversation history
- IMPORTANT: Be consistent with previous responses
- IMPORTANT: If guest refers to previous information, acknowledge it"""
    }
    
    try:
        async_openai_client = openai.AsyncOpenAI()
        
        # Construct messages array with system message, conversation history, and current message
        messages = [system_message]
        
        if conversation_history:
            # Add conversation history, limiting to last 10 messages to prevent token overflow
            history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
            messages.extend(history)
        
        # Add current message
        messages.append({"role": "user", "content": text})
        
        response = await async_openai_client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=messages,
            max_tokens=60,
            temperature=0.7,
            presence_penalty=-0.1,
            frequency_penalty=0.1
        )
        reply = response.choices[0].message.content.strip()
        logger.info(f"GPT reply received. Reply: {reply[:100]}...")
        return reply
    except Exception as e:
        logger.error(f"Error in GPT chat: {e}", exc_info=True)
        return "Sorry, I encountered an error trying to respond."

async def tts_with_elevenlabs(text: str, websocket: WebSocket):
    """Converts text to speech using ElevenLabs API and streams audio to client."""
    if not text:
        logger.warning("tts_with_elevenlabs called with empty text.")
        return

    voice_id = "9q9xpGHwmkXdA4JI72IU"
    model_id = "eleven_multilingual_v2"
    logger.info(f"Requesting TTS from ElevenLabs. Text: {text[:100]}..., Voice: {voice_id}, Model: {model_id}")
    
    try:
        # Get the audio stream using the official streaming method
        audio_stream = elevenlabs_client.text_to_speech.convert_as_stream(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            # stream_chunk_size=1024,
            optimize_streaming_latency=4  # Maximum optimization
        )
        
        logger.info("Starting TTS stream to client")
        
        # Send start streaming marker
        await websocket.send_json({"type": "audio_stream_start"})
        
        # Send audio chunks with appropriate batching for smooth playback
        buffer = io.BytesIO()
        chunk_size = 32768  # Increased for better streaming performance
        chunk_count = 0
        
        for audio_chunk in audio_stream:
            if isinstance(audio_chunk, bytes) and audio_chunk:
                buffer.write(audio_chunk)
                
                # Send chunks when buffer reaches threshold
                if buffer.tell() >= chunk_size:
                    buffer_value = buffer.getvalue()
                    # Base64 encode to avoid binary issues with some WebSocket implementations
                    encoded_chunk = base64.b64encode(buffer_value).decode('utf-8')
                    await websocket.send_json({
                        "type": "audio_chunk", 
                        "chunk": encoded_chunk,
                        "chunk_number": chunk_count
                    })
                    chunk_count += 1
                    # Reset buffer
                    buffer.seek(0)
                    buffer.truncate(0)
        
        # Send any remaining data in the buffer
        if buffer.tell() > 0:
            buffer_value = buffer.getvalue()
            encoded_chunk = base64.b64encode(buffer_value).decode('utf-8')
            await websocket.send_json({
                "type": "audio_chunk", 
                "chunk": encoded_chunk,
                "chunk_number": chunk_count
            })
        
        # Send end streaming marker
        await websocket.send_json({"type": "audio_stream_end"})
        logger.info(f"Completed TTS stream, sent {chunk_count + 1} chunks")
        
        buffer.close()
    except Exception as e:
        logger.error(f"Error in ElevenLabs TTS streaming: {e}", exc_info=True)
        await websocket.send_json({"type": "error", "message": "TTS streaming failed"})

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected: {websocket.client}")

    async def send_json_message(self, message: dict, websocket: WebSocket):
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(message)

    async def send_binary_message(self, data: bytes, websocket: WebSocket):
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_bytes(data)

manager = ConnectionManager()

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    audio_buffer = io.BytesIO()
    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                audio_chunk = data["bytes"]
                audio_buffer.write(audio_chunk)
            elif "text" in data and data["text"] is not None:
                text_payload = data["text"]
                
                try:
                    parsed_json = json.loads(text_payload)
                    if isinstance(parsed_json, dict) and "text" in parsed_json:
                        command = parsed_json["text"]

                        if command == "EOS":
                            logger.info("End of audio stream signal (EOS) received and parsed.")
                            audio_bytes = audio_buffer.getvalue()
                            audio_buffer.seek(0)
                            audio_buffer.truncate(0)

                            if not audio_bytes:
                                logger.warning("Received EOS but audio buffer is empty.")
                                await manager.send_json_message({"type": "error", "message": "No audio received."}, websocket)
                                continue

                            # Start timing
                            start_time = asyncio.get_event_loop().time()

                            logger.info("Processing received audio...")
                            user_transcript = await transcribe_with_whisper(audio_bytes)
                            transcribe_time = asyncio.get_event_loop().time()
                            transcribe_latency = transcribe_time - start_time
                            logger.info(f"Transcription latency: {transcribe_latency:.2f}s")

                            if not user_transcript:
                                 logger.warning("Transcription failed or returned empty.")
                                 await manager.send_json_message({"type": "info", "message": "Could not transcribe audio clearly."}, websocket)
                                 gpt_reply_text = "I couldn't understand what you said. Could you try again?"
                                 await manager.send_json_message({"type": "bot_reply_text", "text": gpt_reply_text}, websocket)
                                 logger.info("Sending TTS for 'could not transcribe' message...")
                                 await tts_with_elevenlabs(gpt_reply_text, websocket)
                                 continue

                            logger.info(f"User transcript: {user_transcript}")
                            await manager.send_json_message({
                                "type": "user_transcript", 
                                "text": user_transcript,
                                "transcribe_latency": transcribe_latency
                            }, websocket)

                            # Get conversation history from websocket state
                            if not hasattr(websocket.state, 'conversation_history'):
                                websocket.state.conversation_history = []
                            
                            # Add user message to history
                            websocket.state.conversation_history.append({"role": "user", "content": user_transcript})

                            logger.info("Getting GPT reply...")
                            gpt_reply_text = await chat_with_gpt(user_transcript, websocket.state.conversation_history)
                            gpt_time = asyncio.get_event_loop().time()
                            gpt_latency = gpt_time - transcribe_time
                            logger.info(f"GPT response latency: {gpt_latency:.2f}s")
                            
                            # Add assistant response to history
                            websocket.state.conversation_history.append({"role": "assistant", "content": gpt_reply_text})
                            
                            logger.info(f"GPT reply: {gpt_reply_text}")
                            await manager.send_json_message({
                                "type": "bot_reply_text", 
                                "text": gpt_reply_text,
                                "gpt_latency": gpt_latency
                            }, websocket)

                            logger.info("Converting GPT reply to speech with ElevenLabs and streaming...")
                            await tts_with_elevenlabs(gpt_reply_text, websocket)
                            end_time = asyncio.get_event_loop().time()
                            total_latency = end_time - start_time
                            logger.info(f"Total latency: {total_latency:.2f}s")
                            
                            # Send latency stats
                            await manager.send_json_message({
                                "type": "latency_stats",
                                "transcribe_latency": transcribe_latency,
                                "gpt_latency": gpt_latency,
                                "total_latency": total_latency
                            }, websocket)

                        elif command == "PING":
                            logger.info("Received PING from client, sending PONG.")
                            await websocket.send_text("PONG")
                        else:
                            logger.warning(f"Received JSON with an unknown inner command: '{command}'. Full payload: {text_payload}")
                    else:
                        logger.warning(f"Received JSON text message, but not in the expected command format: {text_payload}")
                except json.JSONDecodeError:
                    logger.warning(f"Received plain text message (not JSON, and not an expected command structure): {text_payload}")
            else:
                logger.warning(f"Received message with no discernible bytes or text content: {data}")

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {websocket.client}")
        manager.disconnect(websocket)
    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"Client connection closed normally: {websocket.client}")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
             try:
                await manager.send_json_message({"type": "error", "message": f"Server error: {str(e)[:100]}..."}, websocket)
             except Exception as send_e:
                logger.error(f"Failed to send error to client after WebSocket error: {send_e}", exc_info=True)
    finally:
        if audio_buffer and not audio_buffer.closed:
             audio_buffer.close()
        manager.disconnect(websocket)


# Basic HTML frontend for testing
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Audio Chat</title>
    <style>
        * { 
            margin: 0; 
            padding: 0; 
            box-sizing: border-box; 
        }
        
        body { 
            font-family: 'Playfair Display', 'Times New Roman', serif;
            background: url('/static/receptionist.jpg') no-repeat center center fixed;
            background-size: cover;
            min-height: 100vh;
            display: flex;
            align-items: center;
            padding: 40px;
        }

        .container {
            width: 100%;
            max-width: 500px;
            height: 90vh;
            margin-left: 40px;
            background: rgba(255, 255, 255, 0.65);
            backdrop-filter: blur(10px);
            border-radius: 24px;
            box-shadow: 
                0 4px 6px rgba(0, 0, 0, 0.1),
                0 10px 20px rgba(0, 0, 0, 0.05),
                0 20px 40px rgba(0, 0, 0, 0.05);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            position: relative;
            opacity: 0;
            transform: translateY(40px);
            animation: revealCard 1s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        @keyframes revealCard {
            0% {
                opacity: 0;
                transform: translateY(40px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .chat-section {
            flex: 1;
            padding: 40px;
            display: flex;
            flex-direction: column;
            position: relative;
            width: 100%;
            height: 100%;
        }

        .start-call-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            gap: 20px;
        }

        #startCallButton {
            padding: 15px 30px;
            font-size: 18px;
            font-family: 'Playfair Display', serif;
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            opacity: 0;
            transform: translateY(20px);
            animation: revealButton 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards 0.4s;
        }

        #startCallButton:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
            background: linear-gradient(135deg, #45a049, #3d8b40);
        }

        #startCallButton:active {
            transform: translateY(0);
        }

        #startCallButton.hidden {
            display: none;
        }

        .mic-container.hidden {
            display: none;
        }

        @keyframes revealButton {
            0% {
                opacity: 0;
                transform: translateY(20px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .welcome-text {
            font-family: 'Playfair Display', serif;
            font-size: 24px;
            color: #333;
            text-align: center;
            margin-bottom: 20px;
            opacity: 0;
            transform: translateY(20px);
            animation: revealText 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards 0.2s;
        }

        @keyframes revealText {
            0% {
                opacity: 0;
                transform: translateY(20px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .messages-container {
            font-family: 'Poppins', sans-serif;
            letter-spacing: 0.01em;
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            scroll-behavior: smooth;
            height: 100%;
            max-height: calc(90vh - 200px);
            opacity: 0;
            animation: revealMessages 0.8s ease-out forwards 0.8s;
            position: relative;
            -webkit-mask-image: linear-gradient(
                to bottom,
                transparent 0%,
                black 5%,
                black 95%,
                transparent 100%
            );
            mask-image: linear-gradient(
                to bottom,
                transparent 0%,
                black 5%,
                black 95%,
                transparent 100%
            );
        }

        @keyframes revealMessages {
            0% {
                opacity: 0;
            }
            100% {
                opacity: 1;
            }
        }

        .message {
            max-width: 85%;
            padding: 16px 20px;
            border-radius: 16px;
            font-size: 14px;
            line-height: 1.6;
            font-weight: 300;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            opacity: 0;
            transform: translateY(20px);
            animation: messageAppear 0.3s ease-out forwards;
        }

        @keyframes messageAppear {
            0% {
                opacity: 0;
                transform: translateY(20px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .user-message {
            background: rgba(227, 242, 253, 0.9);
            align-self: flex-end;
            margin-left: 15%;
            font-weight: 400;
            transform-origin: bottom right;
            animation: userMessageAppear 0.3s ease-out forwards;
        }

        .bot-message {
            background: rgba(245, 245, 245, 0.9);
            align-self: flex-start;
            margin-right: 15%;
            font-weight: 300;
            transform-origin: bottom left;
            animation: botMessageAppear 0.3s ease-out forwards;
        }

        @keyframes userMessageAppear {
            0% {
                opacity: 0;
                transform: translateX(20px) translateY(20px) scale(0.9);
            }
            100% {
                opacity: 1;
                transform: translateX(0) translateY(0) scale(1);
            }
        }

        @keyframes botMessageAppear {
            0% {
                opacity: 0;
                transform: translateX(-20px) translateY(20px) scale(0.9);
            }
            100% {
                opacity: 1;
                transform: translateX(0) translateY(0) scale(1);
            }
        }

        /* Ensure the chat section takes full height */
        .chat-section {
            flex: 1;
            padding: 40px;
            display: flex;
            flex-direction: column;
            position: relative;
            width: 100%;
            height: 100%;
        }

        /* Hide scrollbar but keep functionality */
        .messages-container::-webkit-scrollbar {
            width: 6px;
        }

        .messages-container::-webkit-scrollbar-track {
            background: transparent;
        }

        .messages-container::-webkit-scrollbar-thumb {
            background-color: rgba(0, 0, 0, 0.1);
            border-radius: 3px;
        }

        .messages-container::-webkit-scrollbar-thumb:hover {
            background-color: rgba(0, 0, 0, 0.2);
        }

        /* Remove the transcript and bot reply areas as they're shown in messages */
        #transcriptArea, #botReplyArea {
            display: none;
        }

        /* Restore the status styles */
        #status {
            font-family: 'Playfair Display', 'Times New Roman', serif;
            font-style: italic;
            text-align: center;
            color: #666;
            font-size: 15px;
            margin-top: 16px;
            min-height: 20px;
            letter-spacing: 0.03em;
            opacity: 0;
            transform: translateY(10px);
            animation: revealStatus 0.6s ease-out forwards 1.5s;
        }

        @keyframes revealStatus {
            0% {
                opacity: 0;
                transform: translateY(10px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Restore the record button styles */
        .controls-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
            position: relative;
        }

        #recordButton,
        #muteButton {
            min-width: 80px;
            height: 36px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.9);
            border: 2px solid rgba(240, 240, 240, 0.8);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            padding: 0 16px;
            font-family: 'Poppins', sans-serif;
            font-size: 14px;
            font-weight: 500;
            color: #666;
            overflow: hidden;
            display: none;
        }

        .button-text {
            transition: all 0.3s ease;
            white-space: nowrap;
        }

        #muteButton .muted-text {
            position: absolute;
            opacity: 0;
            transform: translateY(20px);
        }

        #muteButton.muted {
            background: rgba(255, 75, 75, 0.9);
            border-color: rgba(255, 75, 75, 0.9);
            color: white;
        }

        #muteButton.muted .unmuted-text {
            opacity: 0;
            transform: translateY(-20px);
        }

        #muteButton.muted .muted-text {
            opacity: 1;
            transform: translateY(0);
        }

        #recordButton.recording {
            background: rgba(255, 75, 75, 0.9);
            border-color: rgba(255, 75, 75, 0.9);
            color: white;
        }

        #recordButton:hover,
        #muteButton:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }

        #recordButton:active,
        #muteButton:active {
            transform: scale(0.95);
        }

        .tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
            bottom: -30px;
            white-space: nowrap;
            opacity: 0;
            visibility: hidden;
            transform: translateY(-5px);
            transition: all 0.3s ease;
        }

        .tooltip::before {
            content: '';
            position: absolute;
            top: -4px;
            left: 50%;
            transform: translateX(-50%);
            border-width: 0 4px 4px 4px;
            border-style: solid;
            border-color: transparent transparent rgba(0, 0, 0, 0.8) transparent;
        }

        

        #muteButton:hover .tooltip,
        #recordButton:hover .tooltip {
            opacity: 1;
            visibility: visible;
            transform: translateY(0);
        }

        #muteButton svg,
        #recordButton svg {
            width: 24px;
            height: 24px;
            stroke: #666;
            stroke-width: 2;
            transition: all 0.3s ease;
            position: absolute;
        }

        #muteButton .muted-icon {
            opacity: 0;
            transform: scale(0.8);
        }

        #muteButton.muted {
            background: rgba(255, 75, 75, 0.9);
            border-color: rgba(255, 75, 75, 0.9);
        }

        #muteButton.muted .muted-icon {
            opacity: 1;
            transform: scale(1);
            stroke: white;
        }

        #muteButton.muted .unmuted-icon {
            opacity: 0;
            transform: scale(0.8);
        }

        #muteButton:hover,
        #recordButton:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }

        #muteButton:active,
        #recordButton:active {
            transform: scale(0.95);
        }

        #muteButton:
        display: none;

        #recordButton.recording svg {
            stroke: white;
        }

        .controls-container.hidden {
            display: none;
        }

        #recordButton:hover {
            transform: scale(1.05);
            box-shadow: 
                0 6px 16px rgba(0, 0, 0, 0.15),
                0 10px 20px rgba(0, 0, 0, 0.1);
            border-color: rgba(232, 232, 232, 0.9);
        }

        #recordButton:active {
            transform: scale(0.95);
        }

        /* Talk button styles */
        #recordButton {
            min-width: 120px;
            height: 44px;
            border-radius: 22px;
            background: rgba(76, 175, 80, 0.1);
            border: 2px solid rgba(76, 175, 80, 0.3);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 8px rgba(76, 175, 80, 0.1);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            padding: 0 24px;
            font-family: 'Poppins', sans-serif;
            font-size: 15px;
            font-weight: 500;
            color: #4CAF50;
            overflow: hidden;
        }

        #recordButton .button-text {
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: absolute;
            width: 100%;
            text-align: center;
            opacity: 1;
        }

        #recordButton .initial-text {
            transform: translateY(0);
            opacity: 1;
        }

        #recordButton .recording-text {
            transform: translateY(30px);
            opacity: 0;
            color: white;
        }

        #recordButton .processing-text {
            transform: translateY(30px);
            opacity: 0;
            color: #FFC107;
        }

        #recordButton:hover {
            background: rgba(76, 175, 80, 0.2);
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.2);
            transform: translateY(-1px);
        }

        #recordButton.active-call {
            background: rgba(244, 67, 54, 0.9);
            border-color: #F44336;
            color: white;
            box-shadow: 0 4px 12px rgba(244, 67, 54, 0.2);
        }

        #recordButton.active-call .initial-text {
            transform: translateY(-30px);
            opacity: 0;
        }

        #recordButton.active-call .recording-text {
            transform: translateY(0);
            opacity: 1;
        }

        #recordButton.recording {
            background: rgba(244, 67, 54, 0.9);
            border-color: #F44336;
            color: white;
            animation: recordingPulse 2s infinite;
        }

        @keyframes recordingPulse {
            0% {
                box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.4);
            }
            70% {
                box-shadow: 0 0 0 10px rgba(76, 175, 80, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(76, 175, 80, 0);
            }
        }

        #recordButton.recording::before {
            content: '';
            position: absolute;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.3) 0%, transparent 70%);
            animation: soundWave 1s infinite;
            opacity: 0;
            transform-origin: center;
            --wave-intensity: 0;
        }

        #recordButton.recording::after {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            border-radius: 20px;
            background: linear-gradient(45deg, 
                rgba(76, 175, 80, 0.5), 
                rgba(76, 175, 80, 0.2)
            );
            z-index: -1;
            animation: borderGlow 2s infinite;
            filter: blur(8px);
            opacity: var(--wave-intensity, 0);
        }

        @keyframes soundWave {
            0% {
                transform: scale(1);
                opacity: 0.5;
            }
            100% {
                transform: scale(2);
                opacity: 0;
            }
        }

        @keyframes borderGlow {
            0% {
                opacity: var(--wave-intensity, 0);
                transform: scale(1);
            }
            50% {
                opacity: var(--wave-intensity, 0);
                transform: scale(1.02);
            }
            100% {
                opacity: var(--wave-intensity, 0);
                transform: scale(1);
            }
        }

        #recordButton.processing {
            background: rgba(244, 67, 54, 0.7);
            border-color: #F44336;
            color: white;
        }

        #recordButton.processing .recording-text {
            transform: translateY(-30px);
            opacity: 0;
        }

        #recordButton.processing .processing-text {
            transform: translateY(0);
            opacity: 1;
        }

        #recordButton.processing::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 20px;
            height: 20px;
            margin: -10px 0 0 -10px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            opacity: 0.7;
        }

        @keyframes spin {
            to {
                transform: rotate(360deg);
            }
        }

        @keyframes processingPulse {
            0% {
                box-shadow: 
                    0 4px 12px rgba(255, 193, 7, 0.2),
                    0 8px 16px rgba(255, 193, 7, 0.1),
                    0 0 0 2px rgba(255, 193, 7, 0.2);
                border-color: rgba(255, 193, 7, 0.5);
            }
            50% {
                box-shadow: 
                    0 4px 12px rgba(255, 193, 7, 0.3),
                    0 8px 16px rgba(255, 193, 7, 0.2),
                    0 0 0 4px rgba(255, 193, 7, 0.1);
                border-color: rgba(255, 193, 7, 0.8);
            }
            100% {
                box-shadow: 
                    0 4px 12px rgba(255, 193, 7, 0.2),
                    0 8px 16px rgba(255, 193, 7, 0.1),
                    0 0 0 2px rgba(255, 193, 7, 0.2);
                border-color: rgba(255, 193, 7, 0.5);
            }
        }

        @keyframes processingPulse {
            0% {
                box-shadow: 
                    0 4px 12px rgba(255, 215, 0, 0.2),
                    0 8px 16px rgba(255, 215, 0, 0.1),
                    0 0 0 2px rgba(255, 215, 0, 0.2);
                border-color: rgba(255, 215, 0, 0.5);
            }
            50% {
                box-shadow: 
                    0 4px 12px rgba(255, 215, 0, 0.3),
                    0 8px 16px rgba(255, 215, 0, 0.2),
                    0 0 0 8px rgba(255, 215, 0, 0.1);
                border-color: rgba(255, 215, 0, 0.8);
            }
            100% {
                box-shadow: 
                    0 4px 12px rgba(255, 215, 0, 0.2),
                    0 8px 16px rgba(255, 215, 0, 0.1),
                    0 0 0 2px rgba(255, 215, 0, 0.2);
                border-color: rgba(255, 215, 0, 0.5);
            }
        }

        #recordButton.processing svg {
            fill: rgba(255, 215, 0, 0.9);
            animation: processingIcon 2s infinite cubic-bezier(0.4, 0, 0.2, 1);
        }

        @keyframes processingIcon {
            0% {
                opacity: 0.6;
                transform: scale(0.95);
            }
            50% {
                opacity: 1;
                transform: scale(1.05);
            }
            100% {
                opacity: 0.6;
                transform: scale(0.95);
            }
        }

        /* Add elegant processing indicator */
        #recordButton.processing::after {
            content: '';
            position: absolute;
            top: -4px;
            left: -4px;
            right: -4px;
            bottom: -4px;
            border-radius: 50%;
            border: 2px solid transparent;
            border-top-color: rgba(255, 215, 0, 0.8);
            border-bottom-color: rgba(255, 215, 0, 0.3);
            animation: processingRing 3s infinite ease-in-out;
        }

        @keyframes processingRing {
            0% {
                transform: rotate(0deg) scale(1.02);
                opacity: 0.5;
            }
            50% {
                transform: rotate(180deg) scale(1.05);
                opacity: 0.8;
            }
            100% {
                transform: rotate(360deg) scale(1.02);
                opacity: 0.5;
            }
        }

        #recordButton svg {
            width: 32px;
            height: 32px;
            fill: #666;
            transition: all 0.3s ease;
            transform-origin: center;
        }

        #recordButton:hover svg {
            fill: rgba(255, 75, 75, 0.9);
            transform: scale(1.1);
        }

        #recordButton.recording svg {
            fill: white;
            animation: wave 1.5s infinite ease-in-out;
        }

        @keyframes wave {
            0%, 100% {
                transform: scaleY(1);
            }
            50% {
                transform: scaleY(0.8);
            }
        }

        .ripple {
            position: absolute;
            border-radius: 50%;
            background: rgba(255, 75, 75, 0.2);
            width: 100%;
            height: 100%;
            animation: ripple-animation 1.5s infinite cubic-bezier(0.4, 0, 0.2, 1);
            display: none;
        }

        @keyframes ripple-animation {
            0% {
                transform: scale(0.8);
                opacity: 1;
            }
            100% {
                transform: scale(2);
                opacity: 0;
            }
        }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;1,400&family=Poppins:wght@300;400;500&display=swap" rel="stylesheet">
</head>
<body>
    <div class="container">
        <div class="chat-section">
            <div class="start-call-container" id="startCallContainer">
                <div class="welcome-text">Welcome to the AI Hotel Receptionist</div>
                <button id="startCallButton">Start Call</button>
            </div>
            <div class="messages-container" id="messages"></div>
            <div class="controls-container hidden" id="controlsContainer">
                <button id="muteButton">
                    # <span class="button-text unmuted-text">Unmuted</span>
                    # <span class="button-text muted-text">Muted</span>
                </button>
                <button id="recordButton">
                    <span class="button-text initial-text">Talk</span>
                    <span class="button-text recording-text">End Call</span>
                    <span class="button-text processing-text">Processing...</span>
                    <div class="ripple"></div>
                </button>
            </div>
            <div id="status">Welcome! Click 'Start Call' to begin.</div>
        </div>
    </div>

    <script>
        const startCallButton = document.getElementById('startCallButton');
        const startCallContainer = document.getElementById('startCallContainer');
        const controlsContainer = document.getElementById('controlsContainer');
        const recordButton = document.getElementById('recordButton');
        const muteButton = document.getElementById('muteButton');
        const rippleEffect = recordButton.querySelector('.ripple');
        const statusDisplay = document.getElementById('status');
        const messagesDiv = document.getElementById('messages');
        
        let isMuted = false;

        let websocket;
        let mediaRecorder;
        let isCallActive = false;
        let keepAliveInterval;
        let silenceDetectionTimer;
        let audioContext;
        let audioWorklet;
        let silenceThreshold = -45; // dB
        let silenceDuration = 1500; // ms
        let lastAudioTime = Date.now();
        let isAudioPlaying = false;
        let isSpeaking = false;
        let silenceStartTime = null;
        let conversationHistory = []; // Store conversation history
        
        // Audio streaming and playback variables
        let audioBufferQueue = [];
        let isPlaying = false;
        let audioStreamStarted = false;
        let receivedChunks = [];

        // Function to initialize Web Audio API 
        function initAudioContext() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                console.log("Audio context initialized with sample rate:", audioContext.sampleRate);
            }
        }

        // Function to calculate audio level in dB
        function calculateDB(inputBuffer) {
            const bufferLength = inputBuffer.length;
            let sum = 0;
            
            // Calculate RMS (Root Mean Square)
            for (let i = 0; i < bufferLength; i++) {
                sum += inputBuffer[i] * inputBuffer[i];
            }
            const rms = Math.sqrt(sum / bufferLength);
            
            // Convert to dB
            const db = 20 * Math.log10(rms);
            return db;
        }

        // Enhanced silence detection function with animation
        function detectSilence(audioData) {
            const db = calculateDB(audioData);
            const currentTime = Date.now();
            
            if (db < silenceThreshold) {
                if (!silenceStartTime) {
                    silenceStartTime = currentTime;
                }
                
                if (currentTime - silenceStartTime >= silenceDuration) {
                    if (isSpeaking && mediaRecorder && mediaRecorder.state === "recording") {
                        console.log("Sufficient silence detected after speech, stopping recording");
                        isSpeaking = false;
                        mediaRecorder.stop();
                        
                        // Add processing state with smooth transition
                        recordButton.classList.remove('recording');
                        recordButton.classList.add('processing');
                        statusDisplay.textContent = "Processing...";
                    }
                }
            } else {
                silenceStartTime = null;
                if (!isSpeaking) {
                    isSpeaking = true;
                    recordButton.classList.add('recording');
                    recordButton.classList.remove('processing');
                    statusDisplay.textContent = "Listening...";
                }
                
                // Update animation intensity based on audio level
                const intensity = Math.min(Math.max((db + 60) / 30, 0), 1);
                recordButton.style.setProperty('--wave-intensity', intensity);
                lastAudioTime = currentTime;
            }
        }

        // Modified startNewRecording function
        async function startNewRecording() {
            if (!isCallActive || isAudioPlaying) return;
            
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm; codecs=opus' });
                
                // Reset speech detection states
                isSpeaking = false;
                silenceStartTime = null;
                
                // Apply current mute state to new stream
                stream.getAudioTracks().forEach(track => {
                    track.enabled = !isMuted;
                });
                
                // Set up audio analysis for silence detection
                const audioStream = audioContext.createMediaStreamSource(stream);
                const analyser = audioContext.createAnalyser();
                analyser.fftSize = 2048;
                audioStream.connect(analyser);
                
                const bufferLength = analyser.frequencyBinCount;
                const dataArray = new Float32Array(bufferLength);
                
                function checkAudioLevel() {
                    if (!isCallActive || isAudioPlaying) {
                        if (mediaRecorder && mediaRecorder.state === "recording") {
                            mediaRecorder.stop();
                        }
                        return;
                    }
                    
                    analyser.getFloatTimeDomainData(dataArray);
                    detectSilence(dataArray);
                    if (isCallActive && !isAudioPlaying) {
                        requestAnimationFrame(checkAudioLevel);
                    }
                }
                
                checkAudioLevel();

                mediaRecorder.ondataavailable = (event) => {
                    // Only send audio data if call is active, not muted, not playing audio, and the data is valid
                    if (isCallActive && !isMuted && !isAudioPlaying && event.data.size > 0 && websocket && websocket.readyState === WebSocket.OPEN) {
                        websocket.send(event.data);
                    }
                };

                mediaRecorder.onstart = () => {
                    console.log("Started new recording segment");
                    statusDisplay.textContent = "Listening...";
                };

                mediaRecorder.onstop = () => {
                    // Only process audio if the call is still active
                    if (isCallActive && websocket && websocket.readyState === WebSocket.OPEN && !isAudioPlaying) {
                        // Only send EOS if we detected actual speech followed by silence
                        if (isSpeaking || (Date.now() - lastAudioTime > silenceDuration)) {
                            console.log("Sending audio for transcription");
                            websocket.send(JSON.stringify({text: "EOS"}));
                            statusDisplay.textContent = "Processing...";
                        } else {
                            console.log("No speech detected, starting new recording");
                            startNewRecording();
                        }
                    }
                    // Clean up stream if call is ended
                    if (!isCallActive) {
                        stream.getTracks().forEach(track => track.stop());
                    }
                };

                if (!isAudioPlaying) {
                    mediaRecorder.start(500);
                }
            } catch (err) {
                console.error("Error starting new recording:", err);
                endCall();
            }
        }

        // Function to show the call interface
        function showCallInterface() {
            startCallContainer.style.display = 'none';
            controlsContainer.classList.remove('hidden');
            messagesDiv.style.display = 'flex';
            statusDisplay.textContent = "Click the Talk button to start speaking";
        }

        // Mute button handler
       
        // Function to add message to conversation history
        function addToHistory(role, content) {
            conversationHistory.push({ role, content });
            // Keep only the last 10 messages to prevent context from getting too long
            if (conversationHistory.length > 10) {
                conversationHistory = conversationHistory.slice(-10);
            }
        }

        // Function to reset conversation
        function resetConversation() {
            conversationHistory = [];
            console.log("Conversation history reset");
        }

        // Modified startCall function with smooth transitions
        async function startCall() {
            if (!websocket || websocket.readyState !== WebSocket.OPEN) {
                statusDisplay.textContent = "Connecting to server...";
                connectWebSocket();
                return;
            }

            try {
                initAudioContext();
                isCallActive = true;
                resetConversation(); // Reset conversation history at start of new call
                
                // Smooth transition to active call state
                recordButton.classList.add('active-call');
                rippleEffect.style.display = 'block';
                
                // Update status with slight delay to match animation
                setTimeout(() => {
                    statusDisplay.textContent = "Call in progress... Click 'End Call' to finish.";
                }, 400);
                
                await startNewRecording();
            } catch (err) {
                console.error("Error starting call:", err);
                statusDisplay.textContent = "Error: Could not start call.";
                endCall();
            }
        }

        // Function to end call with smooth transitions
        function endCall() {
            // Set isCallActive to false first to prevent any new recordings
            isCallActive = false;
            
            // Reset speaking state
            isSpeaking = false;
            silenceStartTime = null;
            
            // Clear any pending timers
            if (silenceDetectionTimer) {
                clearTimeout(silenceDetectionTimer);
                silenceDetectionTimer = null;
            }
            
            // Stop media recorder if active
            if (mediaRecorder && mediaRecorder.state !== "inactive") {
                // Remove the ondataavailable handler to prevent sending the last chunk
                mediaRecorder.ondataavailable = null;
                mediaRecorder.stop();
            }
            
            // Stop all audio tracks
            if (mediaRecorder && mediaRecorder.stream) {
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
            
            // Remove all state classes with smooth transition
            recordButton.classList.remove('active-call', 'recording', 'processing');
            rippleEffect.style.display = 'none';
            
            // Reset conversation and update status
            resetConversation();
            statusDisplay.textContent = "Call ended. Click 'Start Call' to begin.";
            recordButton.disabled = false;
            
            // Reset audio processing states
            isAudioPlaying = false;
            audioStreamStarted = false;
            receivedChunks = [];
        }
        
        // Modified processAudioChunk function to update status
        async function processAudioChunk(base64EncodedChunk) {
            try {
                if (mediaRecorder && mediaRecorder.state === "recording") {
                    mediaRecorder.stop();
                }
                isAudioPlaying = true;
                isSpeaking = false;
                silenceStartTime = null;
                statusDisplay.textContent = "AI is speaking...";
                
                const binaryString = atob(base64EncodedChunk);
                const len = binaryString.length;
                const bytes = new Uint8Array(len);
                for (let i = 0; i < len; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                receivedChunks.push(bytes.buffer);
                if (!isPlaying) {
                    playNextChunk();
                }
            } catch (error) {
                console.error('Error processing audio chunk:', error);
                isAudioPlaying = false;
                isSpeaking = false;
                silenceStartTime = null;
            }
        }
        
        // Modified playNextChunk function
        async function playNextChunk() {
            if (receivedChunks.length === 0) {
                isPlaying = false;
                isAudioPlaying = false;
                // Start new recording after audio finishes if call is still active
                if (isCallActive) {
                    statusDisplay.textContent = "Listening...";
                    startNewRecording();
                }
                return;
            }
            isPlaying = true;
            isAudioPlaying = true;
            try {
                const chunk = receivedChunks.shift();
                const audioBuffer = await audioContext.decodeAudioData(chunk);
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContext.destination);
                source.onended = () => {
                    if (receivedChunks.length > 0) {
                        playNextChunk();
                    } else {
                        isPlaying = false;
                        isAudioPlaying = false;
                        // Start new recording after audio finishes if call is still active
                        if (isCallActive) {
                            statusDisplay.textContent = "Listening...";
                            startNewRecording();
                        }
                    }
                };
                source.start(0);
            } catch (error) {
                console.error('Error playing audio chunk:', error);
                isPlaying = false;
                isAudioPlaying = false;
                if (receivedChunks.length > 0) {
                    playNextChunk();
                } else if (isCallActive) {
                    statusDisplay.textContent = "Listening...";
                    startNewRecording();
                }
            }
        }

        function connectWebSocket() {
            const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
            const wsUrl = `${wsProtocol}://${window.location.host}/ws/chat`;
            
            websocket = new WebSocket(wsUrl);

            websocket.onopen = () => {
                console.log("WebSocket connection established.");
                statusDisplay.textContent = "Connected. Click 'Start Call' to begin.";
                startCallButton.disabled = false;
                keepAliveInterval = setInterval(() => {
                    if (websocket && websocket.readyState === WebSocket.OPEN) {
                        websocket.send(JSON.stringify({text: "PING"}));
                    }
                }, 20000);
            };

            websocket.onmessage = async (event) => {
                if (event.data instanceof Blob) {
                    console.log("Received blob data - this format is deprecated");
                } else {
                    try {
                        const message = JSON.parse(event.data);
                        
                        if (message.type === "audio_stream_start") {
                            console.log("Audio stream starting");
                            initAudioContext();
                            receivedChunks = [];
                            isPlaying = false;
                            audioStreamStarted = true;
                            isAudioPlaying = true;
                            if (mediaRecorder && mediaRecorder.state === "recording") {
                                mediaRecorder.stop();
                            }
                            statusDisplay.textContent = "AI is speaking...";
                        }
                        else if (message.type === "audio_chunk") {
                            if (audioStreamStarted) {
                                await processAudioChunk(message.chunk);
                            }
                        }
                        else if (message.type === "audio_stream_end") {
                            console.log("Audio stream complete");
                            audioStreamStarted = false;
                            isAudioPlaying = false;
                            if (isCallActive) {
                                statusDisplay.textContent = "Listening...";
                                startNewRecording();
                            }
                        }
                        else if (message.type === "user_transcript") {
                            console.log(`Transcription completed in ${message.transcribe_latency.toFixed(2)}s`);
                            addToHistory('user', message.text); // Add user message to history
                            addMessage(message.text, 'user');
                        } 
                        else if (message.type === "bot_reply_text") {
                            console.log(`GPT response received in ${message.gpt_latency.toFixed(2)}s`);
                            addToHistory('assistant', message.text); // Add assistant message to history
                            addMessage(message.text, 'bot');
                        } 
                        else if (message.type === "error") {
                            statusDisplay.textContent = `Error: ${message.message}`;
                            console.error("Server error message:", message.message);
                            endCall();
                        } 
                        else if (message.type === "info") {
                            statusDisplay.textContent = message.message;
                        } 
                    } catch (e) {
                        if (event.data === "PONG") {
                            // console.log("Received PONG");
                        } else {
                            console.warn("Received non-JSON message:", event.data);
                        }
                    }
                }
            };

            websocket.onclose = (event) => {
                console.log("WebSocket connection closed:", event.reason, event.code);
                statusDisplay.textContent = "Disconnected. Please refresh.";
                endCall();
                clearInterval(keepAliveInterval);
            };

            websocket.onerror = (error) => {
                console.error("WebSocket error:", error);
                statusDisplay.textContent = "Connection error. Please refresh.";
                endCall();
                clearInterval(keepAliveInterval);
            };
        }
        
        function addMessage(text, sender) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');
            messageDiv.textContent = text;
            messagesDiv.appendChild(messageDiv);
            messageDiv.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }

        // Start call button click handler
        startCallButton.onclick = () => {
            showCallInterface();
        };

        // Record button click handler
        recordButton.onclick = () => {
            if (isCallActive) {
                endCall();
            } else {
                startCall();
            }
        };

        // Initial connection attempt
        connectWebSocket();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get():
    return HTMLResponse(html_template)

if __name__ == "__main__":
    logger.info("Starting Uvicorn server on http://127.0.0.1:8000")
    logger.info("Ensure your .env file is populated with OPENAI_API_KEY and ELEVENLABS_API_KEY.")
    uvicorn.run(app, host="0.0.0.0", port=8000) 
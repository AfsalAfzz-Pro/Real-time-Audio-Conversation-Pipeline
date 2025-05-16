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

async def chat_with_gpt(text: str) -> str:
    """Gets a chatbot reply from OpenAI Chat API."""
    if not text:
        logger.warning("Chat_with_gpt called with empty text.")
        return "I didn't catch that. Could you please repeat?"
    logger.info(f"Sending text to GPT. Text: {text[:100]}...")
    try:
        async_openai_client = openai.AsyncOpenAI()
        response = await async_openai_client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": """You are a professional hotel receptionist at a luxury 5-star hotel. Follow these guidelines:
- Greet guests warmly but efficiently
- Handle check-ins, check-outs, and room inquiries
- Provide information about hotel amenities (pool, spa, restaurant, room service)
- Address guest concerns promptly and professionally
- Keep responses under 30 words and be direct
- Maintain a polite, helpful tone
- If asked about room rates: Standard ($200), Deluxe ($350), Suite ($500)
- Breakfast: 6:30-10:30 AM, Restaurant: 12-11 PM, Pool: 7 AM-9 PM"""},
                {"role": "user", "content": text}
            ],
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

                            logger.info("Getting GPT reply...")
                            gpt_reply_text = await chat_with_gpt(user_transcript)
                            gpt_time = asyncio.get_event_loop().time()
                            gpt_latency = gpt_time - transcribe_time
                            logger.info(f"GPT response latency: {gpt_latency:.2f}s")
                            
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

        .mic-container {
            display: flex;
            justify-content: center;
            margin-top: 20px;
            margin-bottom: 20px;
            opacity: 0;
            transform: scale(0.8);
            animation: revealMic 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards 0.6s;
        }

        @keyframes revealMic {
            0% {
                opacity: 0;
                transform: scale(0.8);
            }
            60% {
                opacity: 1;
                transform: scale(1.1);
            }
            100% {
                opacity: 1;
                transform: scale(1);
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
        #recordButton {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.9);
            border: 2px solid rgba(240, 240, 240, 0.8);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 
                0 4px 12px rgba(0, 0, 0, 0.1),
                0 8px 16px rgba(0, 0, 0, 0.05);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            animation: initialPulse 2s cubic-bezier(0.4, 0, 0.2, 1) forwards 1.2s;
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

        #recordButton.recording {
            background: rgba(255, 75, 75, 0.9);
            border-color: rgba(255, 75, 75, 0.9);
            animation: pulse 2s infinite;
        }

        #recordButton.processing {
            background: rgba(255, 255, 255, 0.95);
            border-color: rgba(255, 215, 0, 0.5);
            animation: processingPulse 2s infinite cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 
                0 4px 12px rgba(255, 215, 0, 0.2),
                0 8px 16px rgba(255, 215, 0, 0.1),
                0 0 0 2px rgba(255, 215, 0, 0.2);
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
            <div class="messages-container" id="messages"></div>
            <div class="mic-container">
            <button id="recordButton">
                    <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 15c2.21 0 4-1.79 4-4V6c0-2.21-1.79-4-4-4S8 3.79 8 6v5c0 2.21 1.79 4 4 4z"/>
                        <path d="M19 11h-1.7c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.49 6-3.31 6-6.72z"/>
                    </svg>
                    <div class="ripple"></div>
            </button>
        </div>
            <div id="status">Click the mic to start recording</div>
        </div>
        <div class="hero-section"></div>
    </div>

    <script>
        const recordButton = document.getElementById('recordButton');
        const rippleEffect = recordButton.querySelector('.ripple');
        const statusDisplay = document.getElementById('status');
        const transcriptArea = document.getElementById('transcriptArea');
        const transcriptText = document.getElementById('transcript');
        const botReplyArea = document.getElementById('botReplyArea');
        const botReplyText = document.getElementById('botReply');
        const messagesDiv = document.getElementById('messages');

        let websocket;
        let mediaRecorder;
        let audioChunks = []; 
        let isRecording = false;
        let keepAliveInterval;
        
        // Audio streaming and playback variables
        let audioContext;
        let audioBufferQueue = [];
        let isPlaying = false;
        let audioStreamStarted = false;
        let receivedChunks = [];

        // Function to initialize Web Audio API 
        function initAudioContext() {
            // Create audio context on user interaction to comply with autoplay policies
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                console.log("Audio context initialized with sample rate:", audioContext.sampleRate);
            }
        }
        
        // Function to decode and play audio chunks as they arrive
        async function processAudioChunk(base64EncodedChunk) {
            try {
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
            }
        }
        
        // Function to play audio chunks sequentially
        async function playNextChunk() {
            if (receivedChunks.length === 0) {
                isPlaying = false;
                return;
            }
            isPlaying = true;
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
                        if (!audioStreamStarted) {
                            resetTTSUI();
                        }
                    }
                };
                source.start(0);
            } catch (error) {
                console.error('Error playing audio chunk:', error);
                isPlaying = false;
                
                // Try the next chunk if this one failed
                if (receivedChunks.length > 0) {
                    playNextChunk();
                }
            }
        }
        
        // Simplified UI reset function
        function resetTTSUI() {
            rippleEffect.style.display = 'none';
            recordButton.classList.remove('processing');
            recordButton.disabled = false;
            statusDisplay.textContent = "Ready. Click mic to record.";
        }

        function connectWebSocket() {
            const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
            const wsUrl = `${wsProtocol}://${window.location.host}/ws/chat`;
            
            websocket = new WebSocket(wsUrl);

            websocket.onopen = () => {
                console.log("WebSocket connection established.");
                statusDisplay.textContent = "Connected. Click mic to record.";
                recordButton.disabled = false;
                recordButton.classList.remove('processing');
                keepAliveInterval = setInterval(() => {
                    if (websocket && websocket.readyState === WebSocket.OPEN) {
                        websocket.send(JSON.stringify({text: "PING"}));
                    }
                }, 20000);
            };

            websocket.onmessage = async (event) => {
                if (event.data instanceof Blob) {
                    console.log("Received blob data - this format is deprecated");
                } else { // JSON data
                    // Try to parse the message as JSON
                    try {
                        const message = JSON.parse(event.data);
                        
                        if (message.type === "audio_stream_start") {
                            console.log("Audio stream starting");
                            initAudioContext();
                            receivedChunks = [];
                            isPlaying = false;
                            audioStreamStarted = true;
                            statusDisplay.textContent = "Bot is speaking...";
                            recordButton.classList.add('processing');
                            rippleEffect.style.display = 'block';
                            recordButton.disabled = true;
                        }
                        else if (message.type === "audio_chunk") {
                            if (audioStreamStarted) {
                                await processAudioChunk(message.chunk);
                            }
                        }
                        else if (message.type === "audio_stream_end") {
                            console.log("Audio stream complete");
                            audioStreamStarted = false;
                            if (!isPlaying) {
                                resetTTSUI();
                            }
                        }
                        else if (message.type === "user_transcript") {
                            console.log(`Transcription completed in ${message.transcribe_latency.toFixed(2)}s`);
                            addMessage(message.text, 'user');
                            transcriptArea.style.display = 'block';
                            transcriptText.textContent = message.text;
                        } 
                        else if (message.type === "bot_reply_text") {
                            console.log(`GPT response received in ${message.gpt_latency.toFixed(2)}s`);
                            addMessage(message.text, 'bot');
                            botReplyArea.style.display = 'block';
                            botReplyText.textContent = message.text;
                        } 
                        else if (message.type === "latency_stats") {
                            console.log("=== Latency Measurements ===");
                            console.log(`Transcription: ${message.transcribe_latency.toFixed(2)}s`);
                            console.log(`GPT Response: ${message.gpt_latency.toFixed(2)}s`);
                            console.log(`Total Processing: ${message.total_latency.toFixed(2)}s`);
                            console.log("========================");
                        }
                        else if (message.type === "error") {
                            statusDisplay.textContent = `Error: ${message.message}`;
                            console.error("Server error message:", message.message);
                            resetTTSUI();
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
                resetTTSUI();
                clearInterval(keepAliveInterval);
            };

            websocket.onerror = (error) => {
                console.error("WebSocket error:", error);
                statusDisplay.textContent = "Connection error. Please refresh.";
                resetTTSUI();
                clearInterval(keepAliveInterval);
            };
        }
        
        function addMessage(text, sender) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');
            messageDiv.textContent = text;
            messagesDiv.appendChild(messageDiv);
            // Scroll to the new message smoothly
            messageDiv.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }

        recordButton.onclick = async () => {
            if (!websocket || websocket.readyState !== WebSocket.OPEN) {
                statusDisplay.textContent = "Connecting to server...";
                connectWebSocket(); 
                return;
            }

            // Initialize audio context on first interaction
            initAudioContext();

            if (isRecording) {
                isRecording = false;
                recordButton.classList.remove('recording');
                rippleEffect.style.display = 'none';
                statusDisplay.textContent = "Processing audio...";
                recordButton.classList.add('processing');
                recordButton.disabled = true;
                if (mediaRecorder && mediaRecorder.state !== "inactive") {
                    mediaRecorder.stop();
                }
            } else {
                // If audio is currently playing, stop it
                if (isPlaying) {
                    // Clear any existing audio processing
                    receivedChunks = [];
                    isPlaying = false;
                    audioStreamStarted = false;
                    resetTTSUI();
                }

                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm; codecs=opus' }); 

                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0) {
                            websocket.send(event.data); 
                        }
                    };
                    
                    mediaRecorder.onstart = () => {
                        isRecording = true;
                        recordButton.classList.add('recording');
                        rippleEffect.style.display = 'block';
                        statusDisplay.textContent = "Recording... Click mic to stop.";
                        transcriptArea.style.display = 'none'; 
                        botReplyArea.style.display = 'none';  
                    };

                    mediaRecorder.onstop = () => {
                        if (websocket && websocket.readyState === WebSocket.OPEN) {
                            websocket.send(JSON.stringify({text: "EOS"}));
                        }
                        stream.getTracks().forEach(track => track.stop());
                    };
                    
                    audioChunks = []; 
                    mediaRecorder.start(500); 
                } catch (err) {
                    console.error("Error accessing microphone:", err);
                    statusDisplay.textContent = "Error: Could not access microphone.";
                    recordButton.classList.remove('recording');
                    recordButton.disabled = false; 
                    rippleEffect.style.display = 'none';
                }
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
    uvicorn.run(app, host="0.0.0.0", port=8000) 
    uvicorn.run(app, host="0.0.0.0", port=8000) 
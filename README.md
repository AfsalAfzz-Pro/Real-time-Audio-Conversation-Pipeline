# Real-time Audio Conversation Pipeline

A high-performance web application demonstrating real-time audio processing and conversation capabilities using modern AI APIs. The system implements a full-duplex audio pipeline with speech-to-text, natural language processing, and text-to-speech capabilities.

## Features

- **Real-time Audio Processing**
  - Browser-based audio capture
  - WebSocket streaming for minimal latency
  - Optimized chunk sizes for audio transmission
  - Efficient audio buffer management

- **Advanced AI Integration**
  - OpenAI Whisper for accurate speech recognition
  - GPT-3.5 Turbo for natural language understanding
  - ElevenLabs for high-quality voice synthesis
  - Sub-4-second round-trip processing

## Technical Stack

- **Backend**
  - FastAPI for async WebSocket handling
  - Python 3.8+ for core processing
  - OpenAI API integration
  - ElevenLabs streaming audio synthesis
  - Async/await architecture for optimal performance

- **Frontend**
  - Pure JavaScript for minimal overhead
  - Web Audio API for audio handling
  - WebSocket for bi-directional communication
  - CSS3 animations and transitions
  - Responsive design principles

## Performance Metrics

- Speech-to-Text: ~1.5s
- Language Processing: ~1.0s
- Text-to-Speech: ~1.5s
- Total Latency: ~4.0s (typical)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=your_openai_key
ELEVENLABS_API_KEY=your_elevenlabs_key
```

4. Run the application:
```bash
python main.py
```

5. Open `http://localhost:8000` in your browser

## Environment Requirements

- Python 3.8+
- Modern web browser with WebSocket support
- Microphone access
- Internet connection for API access

## API Keys Required

- OpenAI API key (for Whisper and GPT-3.5)
- ElevenLabs API key (for voice synthesis)





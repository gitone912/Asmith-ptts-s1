# Install required libraries
# pip install fastapi uvicorn torch transformers soundfile

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
from io import BytesIO
from fastapi.responses import StreamingResponse
# Initialize FastAPI app
app = FastAPI()

# Define CORS origins
origins = [
    "http://localhost:5173",  # Add the origin of your React app
    "http://localhost",        # For other localhost cases
    "*"                        # You can allow all origins by using '*'
]

# Add CORS middleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # You can specify specific origins or use '*'
    allow_credentials=True,
    allow_methods=["*"],    # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers
)

# Device configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

# Request model for TTS input
class TTSRequest(BaseModel):
    prompt: str
    description: str


@app.post("/tts")
async def generate_tts(request: TTSRequest):
    try:
        # Tokenize the input
        input_ids = tokenizer(request.description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(request.prompt, return_tensors="pt").input_ids.to(device)
        
        # Generate the audio
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        
        # Convert tensor to CPU and detach
        audio_arr = generation.cpu().detach().squeeze().tolist()
        
        # Create a bytes buffer to store the audio
        audio_buffer = BytesIO()
        sf.write(audio_buffer, audio_arr, model.config.sampling_rate, format="WAV")
        audio_buffer.seek(0)

        # Return the audio as a streamable response
        return StreamingResponse(audio_buffer, media_type="audio/wav")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating TTS: {str(e)}")
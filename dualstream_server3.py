'''
File: dualstream_server3.py
Created Date: Monday, September 2nd 2024, 8:58:57 pm
Author: alex-crouch

Project Ver 2024
'''

from tts_instream_helper2 import *
from intern_stream_helper import *
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import re

baseAddress = '192.168.193.33'
port = 8000

inference_models = {}
current_task = None

async def stream_text_to_tts(streamer, tts_function):
    buffer = ""
    current_sentence = ""

    for new_text in streamer:
        
        buffer += new_text
        
        # Process complete sentences
        while True:
            match = re.search(r'([^.]*\.[. \n])', buffer)
            if not match:
                break
            
            sentence = match.group(1)
            buffer = buffer[len(sentence):]
            
            # Clean the sentence
            # cleaned_sentence = re.sub(r'[^a-zA-Z0-9$.:\s]', '', sentence).strip()
            cleaned_sentence = sentence

            if cleaned_sentence:
                print(f'{cleaned_sentence}')
                async for audio_chunk in tts_function(cleaned_sentence):
                    yield audio_chunk

    # Print any remaining text in the buffer
    if buffer:
        # cleaned_buffer = re.sub(r'[^a-zA-Z0-9$.:\s]', '', buffer).strip()
        cleaned_buffer = buffer
        if cleaned_buffer:
            print(cleaned_buffer)
            async for audio_chunk in tts_function(cleaned_buffer):
                yield audio_chunk

@asynccontextmanager
async def lifespan(app: FastAPI):
    inference_models['tts'] = TTS_Model()
    inference_models['llm'] = InternModel()
    await inference_models['tts'].warmup()
    yield
    inference_models.clear()
    print('Properly cleaned up')

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def main(file: UploadFile = File(...), text: str = Form(...)):
    image_data = await file.read()
    if image_data:
        print('Received image')
        print(f'Prompt Received: {text}')
        inference_models['llm'].loader(image_data, text)
        
    else:
        raise HTTPException(status_code=400, detail="Invalid image data")
    return StreamingResponse(stream_text_to_tts(inference_models['llm'].streamer, inference_models['tts'].predict))

@app.post("/cancel")
async def cancel():
    global current_task
    if current_task is None:
        return {"message": "No task in progress"}
    
    current_task = None
    inference_models['llm'].cancel()
    inference_models['tts'].cancel()
    return {"message": "Task cancelled successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=baseAddress, port=port)
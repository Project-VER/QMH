'''
File: tts_instream_helper2.py
Created Date: Monday, September 2nd 2024, 9:01:44 pm
Author: alex-crouch

Project Ver 2024
'''

import numpy as np
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager
import scipy.io.wavfile as wavfile
import logging
import os
import asyncio
import uuid

SPEAKER_NAME = "Claribel Dervla"

class TTS_Model:
    def __init__(self, **kwargs):
        self.model = None
        self.speaker = None
        self.is_cancelled = False
        self.current_stream = None
        self.load()

    def load(self):
        device = "cuda"
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        logging.info("⏳Downloading model")
        ModelManager().download_model(model_name)
        model_path = os.path.join(
            get_user_data_dir("tts"), model_name.replace("/", "--")
        )
        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
        self.model.to(device)
        self.speaker = {
            "speaker_embedding": self.model.speaker_manager.speakers[SPEAKER_NAME][
                "speaker_embedding"
            ]
            .cpu()
            .squeeze()
            .half()
            .tolist(),
            "gpt_cond_latent": self.model.speaker_manager.speakers[SPEAKER_NAME][
                "gpt_cond_latent"
            ]
            .cpu()
            .squeeze()
            .half()
            .tolist(),
        }
        logging.info("🔥Model Loaded")

    def wav_postprocess(self, wav):
        """Post process the output waveform"""
        if isinstance(wav, list):
            wav = torch.cat(wav, dim=0)
        wav = wav.clone().detach().cpu().numpy()
        wav = np.clip(wav, -1, 1)
        wav = (wav * 32767).astype(np.int16)
        return wav

    async def predict(self, text):
        self.is_cancelled = False
        language = "en"
        chunk_size = 150
        speaker_embedding = (
            torch.tensor(self.speaker.get("speaker_embedding"))
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        gpt_cond_latent = (
            torch.tensor(self.speaker.get("gpt_cond_latent"))
            .reshape((-1, 1024))
            .unsqueeze(0)
        )
        self.current_stream = self.model.inference_stream(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            stream_chunk_size=chunk_size,
            enable_text_splitting=True,
        )
        try:
            for chunk in self.current_stream:
                if self.is_cancelled:
                    break
                processed_chunk = self.wav_postprocess(chunk)
                processed_bytes = processed_chunk.tobytes()
                await asyncio.sleep(0)
                yield processed_bytes
        finally:
            self.current_stream = None

    def cancel(self):
        self.is_cancelled = True
        if self.current_stream:
            # Attempt to stop the current stream
            try:
                self.current_stream.close()
            except:
                pass  # If close() method is not available or fails, we've at least set is_cancelled
        logging.info("TTS generation cancelled")

    async def warmup(self):
        print("Warming up TTS model...")
        warmup_text = "Welcome to the text-to-speech model warmup. This process ensures that the model is ready for efficient use."

        warmup_audio = b''
        async for audio_chunk in self.predict(warmup_text):
            warmup_audio += audio_chunk
            if len(warmup_audio) > 24000 * 2 * 5:  # ~5 seconds of audio at 24kHz
                break

        # Convert bytes to numpy array
        audio_array = np.frombuffer(warmup_audio, dtype=np.int16)
        
        # Save as WAV file
        output_file = "warmup_audio.wav"
        wavfile.write(output_file, 24000, audio_array)  # Assuming 24kHz sample rate
        
        print(f"Warmup complete. Sample audio saved to {output_file}")
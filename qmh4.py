#!/home/ver/env2/bin/python
'''
File: qmh4.py
Created Date: Wednesday, September 25th 2024, 11:57:47 pm

Project Ver 2024
'''

import os
import time
import requests
from datetime import timedelta

import select
import queue
from queue import Queue, Empty
from collections import deque
from enum import Enum

import asyncio
import threading

import sounddevice as sd
import soundfile as sf
import numpy as np

import gpiod
from gpiod.line import Bias, Edge

import re
from vosk import Model, KaldiRecognizer
import json

from picamera2 import Picamera2, controls
from libcamera import Transform, controls
import cv2
import numpy as np
from PIL import Image

import toml
from threading import Thread, Lock


FALLING_EDGE = "Falling"
RISING_EDGE = "Rising"
GPIO_A = 2
GPIO_B = 3
GPIO_C = 4

IMAGE_WIDTH = 2203
IMAGE_HEIGHT = 1896

class CameraController:
    def __init__(self):
        self.picam2 = Picamera2()
        camera_config = self.picam2.create_video_configuration(
            main={"size": (IMAGE_WIDTH, IMAGE_HEIGHT)},  # 1080p resolution
            #main={"size": (4608, 2592)},  # picam3 max resolution
            controls={"FrameRate": 15}    # Set frame rate to 15 fps for 4K
        )
        self.picam2.configure(camera_config)
        self.picam2.set_controls({"AfMode": 2, "AfRange": 2})
        self.picam2.start()
        time.sleep(2)  # Allow camera to initialize

        self.latest_frame = None
        self.frame_lock = Lock()
        self.running = True
        self.recording_thread = Thread(target=self._record_continuously)
        self.recording_thread.start()

    def _record_continuously(self):
        while self.running:
            frame = self.picam2.capture_array()
            with self.frame_lock:
                self.latest_frame = frame
            time.sleep(1 / 15)  # Adjust based on your desired frame rate

    def capture_image(self):
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            else:
                return None

    def stop(self):
        self.running = False
        self.recording_thread.join()
        self.picam2.stop()

class State(Enum):
    IDLE = 0
    CASEA = 1
    CASEB = 2
    HELP = 3
    BOOT = 4

class AudioStreamer(object):
    def __init__(self, base_address='192.168.193.33', port=8000, samplerate=24000, channels=1):
        self.base_address = base_address
        self.port = port
        self.url = f'http://{base_address}:{port}'
        self.samplerate = samplerate
        self.channels = channels
        self.stream = None
        self.response = None
        self.starttime = None
        self.call = False
        self.waiting_sound, self.file_samplerate = sf.read('/home/ver/cr2/lib_client/waiting.wav')
        self.no_internet_sound, self.file_samplerate = sf.read('/home/ver/cr2/lib_client/internot.wav')
    
    def play_loop(self, no_loop_event, sound):
        time.sleep(2)
        while not no_loop_event.is_set():
            sd.play(sound, self.file_samplerate)
            time.sleep(2)
    
    def run_audio_stream(self, stop_event, image_path, text, timer=False):
        # Flag to print request to stream time handling
        if timer is True:
            self.starttime = time.time()
            self.call = True
        
        # Initialise sounddevice stream
        self.stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype='float32',
        )
        
        files = {'file': open(image_path, 'rb')}
        data = {'text': text}
        
        no_loop_event = threading.Event()
        audioL = threading.Thread(target=self.play_loop, args=(no_loop_event, self.waiting_sound), daemon=True)
        audioL.start()
        try:
            self.response = requests.get(self.url, files=files, data=data, stream=True)

            no_loop_event.set()
            audioL.join()

            with self.stream:
                print("Audio Stream Commenced")
                try:
                    for chunk in self.response.iter_content(chunk_size=56):
                        if self.call is True:   # print time on first chunk if enabled
                            elapsed_time = time.time() - self.starttime
                            print('[{}] finished in {} ms'.format('Request to Stream', int(elapsed_time * 1_000)))
                            self.call = False
                        
                        if chunk and not stop_event.is_set():  # continuously write to output device
                            audio_data = np.frombuffer(chunk, dtype=np.int16)
                            audio_float = audio_data.astype(np.float32) / 32768.0
                            self.stream.write(audio_float)
                        else:
                            print('Cancelled')
                            break
                except Exception as e:
                    print(f"Error: {e}")
                finally:
                    self.response.close()

        except Exception as e:
            print(f"Error: {e}")
            no_loop_event.set()
            audioL.join()
            sd.play(self.no_internet_sound, self.file_samplerate)
            sd.wait()

        finally: # Tidy up stream and connection
            print("Finished Audio Stream")
            self.stream.close()

class SoundPlayer:
    def __init__(self):
        self.sounds = {}
        self.current_frame = 0
        self.data = None

    def load_sound(self, name, file_path):
        sound, sample_rate = sf.read(file_path)
        self.sounds[name] = {"data": sound, "sample_rate": sample_rate}

    def play_sound(self, name):
        if name in self.sounds:
            sd.play(self.sounds[name]["data"], self.sounds[name]["sample_rate"])
        else:
            print(f"Sound '{name}' not found")

    def play_cancellable_sound(self, stop_event, name, *args):
        if name not in self.sounds:
            print(f"Sound '{name}' not found")
            return

        self.data = self.sounds[name]["data"].reshape(-1, 1)
        sample_rate = self.sounds[name]["sample_rate"]

        self.current_frame = 0

        stream = sd.OutputStream(
            samplerate=sample_rate, device=None, channels=1,
            callback=self.output_callback, finished_callback=stop_event.set)
        
        with stream:
            while not stop_event.is_set():
                stop_event.wait(0.1)  # Check every 0.1 seconds

    def output_callback(self, outdata, frames, time, status):
        if status:
            print(status)
        chunk_size = min(len(self.data) - self.current_frame, frames)
        outdata[:chunk_size] = self.data[self.current_frame:self.current_frame + chunk_size]
        if chunk_size < frames:
            outdata[chunk_size:] = 0
            raise sd.CallbackStop()
        self.current_frame += chunk_size

    async def play_sound_async(self, name):
        await asyncio.get_event_loop().run_in_executor(None, self.play_sound, name)

class QueuedMessageHandler:
    def __init__(self):
        self.done_fd = os.eventfd(0)
        self.mq = deque()
        self.state = State.IDLE
        self.ASC = AudioStreamer()
        self.base_address = '192.168.193.33'
        self.port = 8000
        self.cancel_url = f'http://{self.base_address}:{self.port}/cancel'
        self.running = True

        self.sound_player = SoundPlayer()
        self.load_sounds()

        # Load configuration and set up config-related attributes
        self.config = self.load_config()
        self.config_lock = Lock()
        self.update_describe_prompt()

        # Initialise Vosk model for speech recognition
        self.model = Model(lang="en-us")
        self.device_info = sd.query_devices(None, "input")
        self.samplerate = int(self.device_info["default_samplerate"])
        self.rec = KaldiRecognizer(self.model, self.samplerate)
        self.q = queue.Queue()
        
        self.camera_controller = CameraController()
    
    def load_config(self):
        try:
            with open('config.toml', 'r') as f:
                return toml.load(f)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return {
                "prompt": {
                    "input_text": "Use natural language suitable for conversion to speech.",
                    "max_length": 40
                }
            }

    def update_describe_prompt(self):
        with self.config_lock:
            input_prompt = self.config['prompt']['input_text']
            max_length = self.config['prompt']['max_length']
            self.describe_prompt = f'Describe the content in {max_length} words or less. {input_prompt}'

    async def check_config_updates(self):
        while self.running:
            try:
                new_config = self.load_config()
                if new_config != self.config:
                    with self.config_lock:
                        self.config = new_config
                    self.update_describe_prompt()
                    print("Configuration updated")
            except Exception as e:
                print(f"Error checking configuration updates: {e}")
            await asyncio.sleep(5)  # Check every 5 seconds
        
    def load_sounds(self):
        sound_files = {
            "cancel": '/home/ver/cr2/lib_client/cancel.wav',
            "desc": '/home/ver/cr2/lib_client/desc.wav',
            "chat": '/home/ver/cr2/lib_client/chat.wav',
            "helper": '/home/ver/cr2/lib_client/helper.wav',
            "waiting": '/home/ver/cr2/lib_client/waiting.wav',
            "no_internet": '/home/ver/cr2/lib_client/internot.wav'
        }
        for name, file_path in sound_files.items():
            self.sound_player.load_sound(name, file_path)

        self.sound_player.sounds["ping"] = {"data": self.generate_ping(), "sample_rate": 44100}
        # self.sound_player.load_sound("ping", )

    def generate_ping(self, freq1=1500, freq2=1800, duration=0.15, sample_rate=44100):
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Generate two sine waves
        tone1 = np.sin(2 * np.pi * freq1 * t)
        tone2 = np.sin(2 * np.pi * freq2 * t)
        
        # Combine the tones
        tone = tone1 + tone2
        
        # Normalize
        tone = tone / np.max(np.abs(tone))
        
        # Apply an exponential decay envelope
        envelope = np.exp(-t * 30)
        tone = tone * envelope
        
        return tone

    def input_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.q.put(bytes(indata))

    def edge_type_str(self, event):
        if event.event_type is event.Type.RISING_EDGE:
            return RISING_EDGE
        if event.event_type is event.Type.FALLING_EDGE:
            return FALLING_EDGE
        return "Unknown"

    async def async_watch_line_value(self, chip_path, line_offsets, done_fd):
        with gpiod.request_lines(
            chip_path,
            consumer="gpio-state-machine",
            config={tuple(line_offsets): gpiod.LineSettings(edge_detection=Edge.BOTH, debounce_period=timedelta(milliseconds=20))},
        ) as request:
            while self.running:
                # Use asyncio.get_event_loop().run_in_executor to run blocking operations in a separate thread
                events = await asyncio.get_event_loop().run_in_executor(None, request.read_edge_events)
                for event in events:
                    self.mq.append([event.line_offset, self.edge_type_str(event)])
                
                # Check if we need to stop
                if await self.is_done(done_fd):
                    return

    async def is_done(self, done_fd):
        # Use asyncio.select to check the done_fd without blocking
        r, _, _ = await asyncio.get_event_loop().run_in_executor(None, select.select, [done_fd], [], [], 0)
        return bool(r)

    def input_speech(self, stop_event):
        detection = None
        with sd.RawInputStream(samplerate=self.samplerate, blocksize = 8000, device=None,
            dtype="int16", channels=1, callback=self.input_callback): # begin recording
            print("Start speaking...")
            rec = KaldiRecognizer(self.model, self.samplerate)
            while not stop_event.is_set():
                data = self.q.get() # Get audio data from the queue
                if rec.AcceptWaveform(data):
                    self.sound_player.play_sound('ping') # Play acknowledgement ping
                    ting = json.loads(rec.Result()) # Inference on model
                    # print(ting.get("text", "")) # Format detection
                    return ting.get("text", "") # Format detection

    async def audio_threader(self, target_function, input_cancel = False,
        target_sound = None, image_path = None, prompt = None, que = None):
        stop_event = threading.Event()

        if prompt is not None:
            arg_list = (stop_event, image_path, prompt)
        elif input_cancel:
            arg_list = (que, stop_event)
        else:
            arg_list = (stop_event, target_sound)

        audio_thread = threading.Thread(
            target=target_function, 
            args=arg_list, 
            daemon=True
        )
        audio_thread.start()
        if input_cancel or prompt is not None:
            await asyncio.sleep(1)
            self.mq.clear()
        while audio_thread.is_alive():
            await asyncio.sleep(0.1)
            if self.mq.count([GPIO_A, FALLING_EDGE]) != 0 or self.mq.count([GPIO_B, FALLING_EDGE]) != 0:
                stop_event.set()
                if prompt is not None:
                    try:
                        response = requests.post(self.cancel_url)
                    except Exception as e:
                        print(f"Error: {e}")
                    await asyncio.sleep(0.1)  # Give a short time for the audio stream to stop
                    await self.sound_player.play_sound_async('cancel')
                    audio_thread.join()
                    self.mq.clear()
                    break
                elif input_cancel:
                    await self.sound_player.play_sound_async('cancel')
                else:
                    await asyncio.sleep(0.1)
                    break
        try:
            audio_thread.join()
        except:
            pass
        self.mq.clear()

    def rotate_and_save(self, pre_image):
        rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        image_path = 'best_image_q_pre.png'
        cv2.imwrite(image_path, rgb)
        rgb = Image.fromarray(rgb)
        post_image = rgb.rotate(30,expand=True)
        post_image = np.array(post_image)
        #rotate_and_save
        #crop
        image_path = 'best_image_q.png'
        cv2.imwrite(image_path, post_image)

        return post_image

    async def state_run(self):
        while self.running:
            await asyncio.sleep(0.01)

            if self.state == State.IDLE:
                await asyncio.sleep(0.01)
                if self.mq.count([GPIO_A, FALLING_EDGE])!=0:
                    self.state = State.CASEA
                if self.mq.count([GPIO_B, FALLING_EDGE])!=0:
                    self.state = State.CASEB
                if self.mq.count([GPIO_C, FALLING_EDGE])!=0:
                    self.state = State.HELP

            elif self.state == State.CASEA:
                print('CASE A')
                image_array = self.camera_controller.capture_image()
                self.rotate_and_save(image_array)
                image_path = 'best_image_q.png'
                asyncio.create_task(self.sound_player.play_sound_async('desc'))
                
                
                await self.audio_threader(self.ASC.run_audio_stream, input_cancel = False,
                    target_sound = None, image_path = image_path,
                    prompt = self.describe_prompt)

                self.state = State.IDLE
            
            elif self.state == State.CASEB:
                print('CASE B')
                image_array = self.camera_controller.capture_image()
                self.rotate_and_save(image_array)
                image_path = 'best_image_q.png'
                
                asyncio.create_task(self.sound_player.play_sound_async('chat'))

                result = None
                que = Queue()
                
                await self.audio_threader(target_function = lambda q, arg1: q.put(self.input_speech(arg1)), input_cancel = True,
                    target_sound = None, image_path = None, prompt = None, que = que)

                result = que.get()
                print(result)

                if result != None:
                    await self.audio_threader(self.ASC.run_audio_stream, input_cancel = False,
                        target_sound = None, image_path = image_path,
                        prompt = result)
                    self.state = State.IDLE
                else:
                    await self.sound_player.play_sound_async('cancel')
                    self.mq.clear()
                    self.state = State.IDLE
            
            elif self.state == State.HELP:
                print('Help')
                await asyncio.create_task(self.sound_player.play_sound_async('helper'))
                #await self.audio_threader(self.sound_player.play_sound_async, input_cancel = False,
                #    target_sound = 'helper', image_path = None,
                #    prompt = None)
                self.mq.clear()
                self.state = State.IDLE

    async def run(self):
        await asyncio.gather(
            self.async_watch_line_value("/dev/gpiochip4", [GPIO_A, GPIO_B, GPIO_C], self.done_fd),
            self.state_run(),
            self.check_config_updates())

    def stop(self):
        self.running = False

async def main():
    qmh = QueuedMessageHandler()
    try:
        await qmh.run()
    except asyncio.CancelledError:
        qmh.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopping...")

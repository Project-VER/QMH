'''
File: intern_stream_helper.py
Created Date: Friday, July 26th 2024, 9:07:03 pm
Author: alex-crouch

Project Ver 2024
'''

import torch
import torchvision.transforms as T
from PIL import Image
import io
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import wave

class InternModel:
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, path='OpenGVLab/InternVL2-8B', max_num=6, input_size=448):
        self.pixel_values = None
        self.question = None
        self.path = path
        self.max_num = max_num
        self.input_size = input_size
        self.model, self.tokenizer, self.generation_config, self.streamer = self.load_intern()

    @staticmethod
    def build_transform(input_size):
        MEAN, STD = InternModel.IMAGENET_MEAN, InternModel.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    @staticmethod
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j) for n in range(min_num, self.max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= self.max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, self.input_size)

        target_width = self.input_size * target_aspect_ratio[0]
        target_height = self.input_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // self.input_size)) * self.input_size,
                (i // (target_width // self.input_size)) * self.input_size,
                ((i % (target_width // self.input_size)) + 1) * self.input_size,
                ((i // (target_width // self.input_size)) + 1) * self.input_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((self.input_size, self.input_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file):
        if isinstance(image_file, str):
            image = Image.open(image_file).convert('RGB')
        elif isinstance(image_file, bytes):
            image = Image.open(io.BytesIO(image_file)).convert('RGB')
        else:
            raise ValueError("Unsupported image input type")
        
        transform = self.build_transform(input_size=self.input_size)
        images = self.dynamic_preprocess(image, use_thumbnail=True)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    # initialises the LLM model
    def load_intern(self):
        model = AutoModel.from_pretrained(
            self.path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().cuda()

        tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)

        # Initialize the streamer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10)
        # Define the generation configuration
        generation_config = dict(num_beams=1, max_new_tokens=1024, do_sample=False, streamer=streamer)

        return model, tokenizer, generation_config, streamer

    # loads the image and prompt into the LLM so that it is ready when called
    def loader(self, image_path, text):
        pixel_values = self.load_image(image_path).to(torch.bfloat16).cuda()
        mode = '<image>\n'
        actualq = mode + text
        # Start the model chat in a separate thread
        thread = Thread(target=self.model.chat, kwargs=dict(
            tokenizer=self.tokenizer, pixel_values=pixel_values, question=actualq,
            history=None, return_history=False, generation_config=self.generation_config,
        ))
        thread.start()

        # ADDED!!!
        # Initialize an empty string to store the generated text
        generated_text = ''
        # Loop through the streamer to get the new text as it is generated
        for new_text in self.streamer:
            if new_text == self.model.conv_template.sep:
                break
            generated_text += new_text
            print(new_text, end='', flush=True)  # Print each new chunk of generated text on the same line
        

    def cancel(self):
        if hasattr(self, 'streamer'):
            self.streamer.cancel()
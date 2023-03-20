import argparse
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipConfig



def caption_image(image_path):
    raw_image = Image.open(image_path).convert('RGB')

    # conditional image captioning
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt")

    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True, )

def caption_unconditional(image_path):
    raw_image = Image.open(image_path).convert('RGB')

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True, )

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

if __name__ == "__main__":
    # get cli args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="Path to the image folder.", required=True)
    images = [os.path.join(parser.parse_args().image, image) for image in os.listdir(parser.parse_args().image) if image.endswith(".jpg")]
    for x  in images:
        print(caption_image(images[0]))



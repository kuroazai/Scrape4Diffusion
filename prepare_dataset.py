import os
from PIL import Image
import config as cfg
from machine_learning import blip_image_caption


def generate_lora_folder_structure(folder: str):
    if not os.path.exists(os.path.join(folder, "model")):
        os.mkdir(os.path.join(folder, "model"))
    if not os.path.exists(os.path.join(folder, "image")):
        os.mkdir(os.path.join(folder, "image"))
    if not os.path.exists(os.path.join(folder, "log")):
        os.mkdir(os.path.join(folder, "log"))


def resize_image(image_path, resolution):
    """
    Resizes an image based on the input resolution and saves it to the same directory.

    Args:
        image_path (str): Path to the image file.
        resolution (tuple): Tuple of (width, height) in pixels.

    Returns:
        None
    """
    with Image.open(image_path) as img:
        return  img.resize(resolution)


def resize_images(images: list, output_dir: str):
    for i,image in enumerate(images):
        resized_img = resize_image(image, (cfg.resolution, cfg.resolution))
        caption_text = blip_image_caption.caption_image(image)
        resized_img.save(os.path.join(image_dir, image.split("\\")[-1]))
        print(image)
        caption_file = image.split('\\')[-1].split('.')[0] + ".txt"
        print(caption_file)
        with open(os.path.join(image_dir, caption_file), "w") as file:
            file.write(caption_text)
            file.close()
        os.remove(image)

if __name__ == '__main__':
    outdir = os.path.join(os.getcwd(), f"output/")
    for folder in os.listdir(outdir):
        path = os.path.join(outdir, folder)
        generate_lora_folder_structure(path)
        image_dir = os.path.join(path, f"image/{cfg.repeats}_{folder}")
        if os.path.exists(image_dir):
            pass
        else:
            os.mkdir(image_dir)
    
        generate_lora_folder_structure(path)
        images = [os.path.join(path, image) for image in os.listdir(path) if image.endswith(".jpg")]
        resize_images(images, image_dir)


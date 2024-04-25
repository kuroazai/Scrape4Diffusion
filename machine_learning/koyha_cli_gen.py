import argparse
import os
import config as cfg

model_base = 'model name'
train_on = 'model path' + model_base
stagin_folder = ''
image_folder = ''
def get_models(search=False):
    dirs = os.listdir(stagin_folder)
    model_dict = {}
    if search:
        image_folder = os.path.join(stagin_folder, parser.parse_args().model_name, f'image/{cfg.repeats}_{parser.parse_args().model_name}')
        main_im_fldr = f'{stagin_folder}/{parser.parse_args().model_name}/image/'
        model_folder = f'{stagin_folder}/{parser.parse_args().model_name}/model/'
        log_folder = f'{stagin_folder}/{parser.parse_args().model_name}/log/'
        images = len(os.listdir(image_folder))
        model_dict = {**model_dict, **build_command(parser.parse_args().model_name, main_im_fldr, model_folder, log_folder, cfg.resolution, cfg.repeats, images)}
    else:
        for model in dirs:
            image_folder = os.path.join(stagin_folder, model, f'image/{cfg.repeats}_{model}')
            main_im_fldr = f'{stagin_folder}/{model}/image/'
            model_folder = f'{stagin_folder}/{model}/model/'
            log_folder = f'{stagin_folder}/{model}/log/'
            if os.path.exists(os.path.join(model_folder, f'{model}.safetensors')):
                print(f"Skipping {model} because has already been processed")
                continue
            cmd = f'.\venv\Scripts\python.exe "finetune/make_captions.py" --batch_size="3" --num_beams="5" --top_p="0.9" --max_length="75" --min_length="5" --beam_search --caption_extension=".txt" "{stagin_folder}/{model}/image/15_{model}" --caption_weights="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth"'
            print("Captioning images...")
            os.system(cmd)
            images = len(os.listdir(image_folder))
            model_dict = {**model_dict, **build_command(model, main_im_fldr, model_folder, log_folder, cfg.resolution, cfg.repeats, images)}

    return model_dict

def build_command(model_name: str, image_folder: str, model_folder: str, log_folder: str, res: int, repeats: int, images: int) -> dict:
    max_train_steps = images * repeats
    command = f'accelerate launch --num_cpu_threads_per_process=2 "train_network.py" --enable_bucket --pretrained_model_name_or_path="{train_on}" --train_data_dir="{image_folder}" --resolution={res},{res} --output_dir="{model_folder}" --network_alpha="1" --save_model_as=safetensors --network_module=networks.lora --text_encoder_lr=5e-5 --unet_lr=0.0001 --network_dim=8 --output_name="{model_name}" --lr_scheduler_num_cycles="3" --learning_rate="0.0001" --lr_scheduler="cosine" --lr_warmup_steps="324" --train_batch_size="1" --max_train_steps="{max_train_steps}" --save_every_n_epochs="1" --mixed_precision="fp16" --save_precision="fp16" --seed="1234" --optimizer_type="AdamW" --bucket_reso_steps=64 --mem_eff_attn --gradient_checkpointing --bucket_no_upscale'
    print("training model...")
    os.system(command)
    return {model_name: command}

def main():
    data = get_models().get(model)
    print(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        default='', required=False)
    model = parser.parse_args().model_name
    main()

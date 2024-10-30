import random
from sample import (arg_parse, 
                    sampling,
                    load_fontdiffuer_pipeline)
import os
import time

def fetch_lantingjixu_chars():
    with open('lantingjixu.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()
        characters = list(set(text))
        return characters

#called by main function of lantingjixu_samople
def run_fontdiffuer(content_image_path, 
                    character, 
                    style_image_path,
                    save_image_dir,
                    sampling_step,
                    guidance_scale,
                    batch_size,
                    seed):
    args.demo = False
    args.content_image_path = content_image_path
    args.style_image_path = style_image_path
    args.save_image_dir = save_image_dir
    args.character_input = False if content_image_path is not None else True
    args.content_character = character
    args.sampling_step = sampling_step
    args.guidance_scale = guidance_scale
    args.batch_size = batch_size
    args.seed = seed if type(seed) is int else random.randint(0, 10000)
    out_image = sampling(
        args=args,
        pipe=pipe,      # use the loaded fontdiffuer pipeline with the model of FontDiffuserModelDPM
        content_image=content_image_path,
        style_images=style_image_path)
    return out_image

if __name__ == '__main__':
    args = arg_parse()
    args.ckpt_dir = 'ckpt/'
    args.ttf_path = 'ttf/SourceHanSerifTC-VF.ttf'

    args.method = 'multistep'
    args.guidance_type = 'classifier-free'
    args.algorithm_type = 'dpmsolver++'
    args.device = "cpu"
    args.style_image_path = "./data_examples/style_images"
    # load fontdiffuer pipeline
    pipe = load_fontdiffuer_pipeline(args=args)

    # load lantingjixu sample
    characters = fetch_lantingjixu_chars()
    total_time = 0      
    total_sample = 0        

    no_existence_check = False      # set to True to skip the existence check

    for i, character in enumerate(characters):
        if not no_existence_check and os.path.exists(f'outputs/{character}.png'):
            print(f'[{i+1}/{len(characters)}] outputs/{character}.png already exists')
        else:
            start_time = time.time()
            # run fontdiffuer
            out_image = run_fontdiffuer(content_image_path=None,
                                        character=character,
                                        style_image_path='data_examples/sampling',
                                        save_image_dir='outputs/',
                                        sampling_step=20,
                                        guidance_scale=30,
                                        batch_size=1,
                                        seed=0)
            end_time = time.time()
            print(f"Finish the sampling process, costing time {end_time - start_time}s")
            total_time += end_time - start_time
            total_sample += 1
            out_image.save(f'outputs/{character}.png')
            print(f'[{i+1}/{len(characters)}] created outputs/{character}.png')

    print(f"Total sampling time: {total_time}s")
    print(f"Total sampling: {total_sample}")
    print(f"Average sampling time: {total_time/total_sample}s")
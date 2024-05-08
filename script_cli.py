from PIL import Image
from flashface.all_finetune.test_script import generate
import os
from contextlib import contextmanager
import argparse

@contextmanager
def clear_cache_and_gpu():
    try:
        yield
    finally:
        import torch
        torch.cuda.empty_cache()

# hyparams here
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str)
parser.add_argument("--save_file_name", type=str)
parser.add_argument("--face_img_1", type=str, default='')
parser.add_argument("--face_img_2", type=str, default='')
parser.add_argument("--face_img_3", type=str, default='')
parser.add_argument("--face_img_4", type=str, default='')
parser.add_argument("--need_detect", action='store_true', default=True)
parser.add_argument("--a_prompt", type=str,
                    default='best quality, masterpiece, ultra-detailed, UHD 4K, photographic')
parser.add_argument("--n_prompt", type=str,
                    default='blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, '
                            'out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, '
                            'signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--face_bbox_0", type=float, default=0.0)
parser.add_argument("--face_bbox_1", type=float, default=0.0)
parser.add_argument("--face_bbox_2", type=float, default=0.0)
parser.add_argument("--face_bbox_3", type=float, default=0.0)
parser.add_argument("--lamda_feat", type=float, default=0.9)
parser.add_argument("--face_guidence", type=float, default=2.5)
parser.add_argument("--step_to_launch_face_guidence", type=int, default=700)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--text_control_scale", type=float, default=7.5)
parser.add_argument("--seed", type=int, default=-1)
args = parser.parse_args()
print(args)

face_img_names = [args.face_img_1, args.face_img_2, args.face_img_3, args.face_img_4]
# Убедитесь, что в списке есть хотя бы одно изображение
if all(name == '' for name in face_img_names):
    raise argparse.ArgumentTypeError('At least one face image must be provided')

face_imgs = [Image.open(name).convert("RGB") for name in face_img_names if name != '']

# face position
face_bbox = [args.face_bbox_0, args.face_bbox_1, args.face_bbox_2, args.face_bbox_3]

print('# start generate images')
with clear_cache_and_gpu():
    imgs = generate(
        pos_prompt=args.a_prompt,
        neg_prompt=args.n_prompt,
        steps=args.steps,
        face_bbox=face_bbox,
        lamda_feat=args.lamda_feat,
        face_guidence=args.face_guidence,
        num_sample=args.batch_size,
        text_control_scale=args.text_control_scale,
        seed=args.seed,
        step_to_launch_face_guidence=args.step_to_launch_face_guidence,
        reference_faces=face_imgs,
        need_detect=args.need_detect
    )

# show the generated images
print('generated!')
for i, img in enumerate(imgs):
    img.save(f'{args.save_dir}/{args.save_file_name}_{i}.png')

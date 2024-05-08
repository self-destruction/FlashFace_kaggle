import sys
import os
package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, package_dir)

import copy
import random
import numpy as np

from contextlib import contextmanager

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torchvision.transforms as T
from flashface.all_finetune.config import cfg
from PIL import Image, ImageDraw

from flashface.all_finetune.models import sd_v1_ref_unet
from flashface.all_finetune.ops.context_diffusion import ContextGaussianDiffusion

from ldm import data, models, ops
from ldm.models.vae import sd_v1_vae

import torchvision.transforms as T
from flashface.all_finetune.utils import Compose, PadToSquare, get_padding, seed_everything
from ldm.models.retinaface import retinaface, crop_face

# model path
SKIP_LOAD = False
DEBUG_VIEW = False
SKEP_LOAD = False
LOAD_FLAG = True
DEFAULT_INPUT_IMAGES = 4
MAX_INPUT_IMAGES = 4
SIZE = 768
with_lora = False
enable_encoder = False
with_pos_mask = True

weight_path = f'{package_dir}/cache/flashface.ckpt'

gpu = 'cuda'

padding_to_square = PadToSquare(224)

retinaface_transforms = T.Compose([PadToSquare(size=640), T.ToTensor()])

rf = retinaface(pretrained=True, device=gpu).eval().requires_grad_(False)

@contextmanager
def clear_cache_and_gpu():
    try:
        yield
    finally:
        unet.share_cache.clear()
        torch.cuda.empty_cache()

def detect_face(imgs=None):
    # read images
    pil_imgs = imgs
    b = len(pil_imgs)
    vis_pil_imgs = copy.deepcopy(pil_imgs)

    # detection
    imgs = torch.stack([retinaface_transforms(u) for u in pil_imgs]).to(gpu)
    boxes, kpts = rf.detect(imgs, min_thr=0.6)

    # undo padding and scaling
    face_imgs = []

    for i in range(b):
        # params
        scale = 640 / max(pil_imgs[i].size)
        left, top, _, _ = get_padding(round(scale * pil_imgs[i].width),
                                      round(scale * pil_imgs[i].height), 640)

        # undo padding
        boxes[i][:, [0, 2]] -= left
        boxes[i][:, [1, 3]] -= top
        kpts[i][:, :, 0] -= left
        kpts[i][:, :, 1] -= top

        # undo scaling
        boxes[i][:, :4] /= scale
        kpts[i][:, :, :2] /= scale

        # crop faces
        crops = crop_face(pil_imgs[i], boxes[i], kpts[i])
        if len(crops) != 1:
            raise ValueError(
                f'Warning: {len(crops)} faces detected in image {i}')

        face_imgs += crops

        # draw boxes on the pil image
        draw = ImageDraw.Draw(vis_pil_imgs[i])
        for box in boxes[i]:
            box = box[:4].tolist()
            box = [int(x) for x in box]
            draw.rectangle(box, outline='red', width=4)

    face_imgs = face_imgs

    return face_imgs





def encode_text(m, x):
    # embeddings
    x = m.token_embedding(x) + m.pos_embedding

    # transformer
    for block in m.transformer:
        x = block(x)

    # output
    x = m.norm(x)

    return x


def generate(
    pos_prompt,
    neg_prompt,
    steps=35,
    face_bbox=[0., 0., 0., 0.],
    lamda_feat=0.9,
    face_guidence=2.2,
    num_sample=1,
    text_control_scale=7.5,
    seed=0,
    step_to_launch_face_guidence=600,
    lamda_feat_before_ref_guidence=0.85,
    reference_faces=None,
    need_detect=True,
    default_pos_prompt='best quality, masterpiece, ultra-detailed, UHD 4K, photographic',
    default_neg_prompt='blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face',
):
    if not DEBUG_VIEW and not SKEP_LOAD:
        clip_tokenizer = data.CLIPTokenizer(padding='eos')
        clip = getattr(models, cfg.clip_model)(
            pretrained=True).eval().requires_grad_(False).textual.to(gpu)
        autoencoder = sd_v1_vae(
            pretrained=True).eval().requires_grad_(False).to(gpu)
    
        unet = sd_v1_ref_unet(pretrained=True,
                              version='sd-v1-5_nonema',
                              enable_encoder=enable_encoder).to(gpu)
    
        unet.replace_input_conv()
        unet = unet.eval().requires_grad_(False).to(gpu)
        unet.share_cache['num_pairs'] = cfg.num_pairs
    
    
        if LOAD_FLAG:
            model_weight = torch.load(weight_path, map_location="cpu")
            msg = unet.load_state_dict(model_weight, strict=True)
            print(msg)
    
        # diffusion
        sigmas = ops.noise_schedule(schedule=cfg.schedule,
                                    n=cfg.num_timesteps,
                                    beta_min=cfg.scale_min,
                                    beta_max=cfg.scale_max)
        diffusion = ContextGaussianDiffusion(sigmas=sigmas,
                                             prediction_type=cfg.prediction_type)
        diffusion.num_pairs = cfg.num_pairs
        print("model initialized")
    
    face_transforms = Compose(
        [T.ToTensor(),
         T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    
    
    solver = 'ddim'
    # solver = cfg.solver
    if default_pos_prompt is not None:
        pos_prompt = pos_prompt + ', ' + default_pos_prompt
    if neg_prompt is not None:
        neg_prompt = neg_prompt + ', ' + default_neg_prompt
    else:
        neg_prompt = default_neg_prompt
    if seed == -1:
        seed = random.randint(0, 2147483647)
    seed_everything(seed)

    print(seed)
    print('final pos_prompt: ', pos_prompt)
    print('final neg_prompt: ', neg_prompt)

    if need_detect:
        reference_faces = detect_face(reference_faces)

        # for i, ref_img in enumerate(reference_faces):
        #     ref_img.save(f'./{i + 1}.png')
        print(f'detected {len(reference_faces)} faces')
        assert len(
            reference_faces) > 0, 'No face detected in the reference images'

        if len(reference_faces) < 4:
            expand_reference_faces = copy.deepcopy(reference_faces)
            while len(expand_reference_faces) < 4:
                # random select from ref_imgs
                expand_reference_faces.append(random.choice(reference_faces))
            reference_faces = expand_reference_faces

    # process the ref_imgs
    H = W = SIZE

    normalized_bbox = face_bbox
    print(normalized_bbox)
    face_bbox = [
        int(normalized_bbox[0] * W),
        int(normalized_bbox[1] * H),
        int(normalized_bbox[2] * W),
        int(normalized_bbox[3] * H)
    ]
    max_size = max(face_bbox[2] - face_bbox[1], face_bbox[3] - face_bbox[1])
    empty_mask = torch.zeros((H, W))

    empty_mask[face_bbox[1]:face_bbox[1] + max_size,
               face_bbox[0]:face_bbox[0] + max_size] = 1

    empty_mask = empty_mask[::8, ::8].cuda()
    empty_mask = empty_mask[None].repeat(num_sample, 1, 1)

    pasted_ref_faces = []
    show_refs = []
    for ref_img in reference_faces:
        ref_img = ref_img.convert('RGB')
        ref_img = padding_to_square(ref_img)
        to_paste = ref_img

        to_paste = face_transforms(to_paste)
        pasted_ref_faces.append(to_paste)

    faces = torch.stack(pasted_ref_faces, dim=0).to(gpu)

    c = encode_text(clip, clip_tokenizer([pos_prompt]).to(gpu))
    c = c[None].repeat(num_sample, 1, 1, 1).flatten(0, 1)
    c = {'context': c}

    single_null_context = encode_text(clip,
                                      clip_tokenizer([neg_prompt
                                                      ]).cuda()).to(gpu)
    null_context = single_null_context
    nc = {
        'context': null_context[None].repeat(num_sample, 1, 1,
                                             1).flatten(0, 1)
    }

    ref_z0 = cfg.ae_scale * torch.cat([
        autoencoder.sample(u, deterministic=True)
        for u in faces.split(cfg.ae_batch_size)
    ])
    #  ref_z0 = ref_z0[None].repeat(num_sample, 1,1,1,1).flatten(0,1)
    with clear_cache_and_gpu():
        unet.share_cache['num_pairs'] = 4
        unet.share_cache['ref'] = ref_z0
        unet.share_cache['similarity'] = torch.tensor(lamda_feat).cuda()
        unet.share_cache['ori_similarity'] = torch.tensor(lamda_feat).cuda()
        unet.share_cache['lamda_feat_before_ref_guidence'] = torch.tensor(
            lamda_feat_before_ref_guidence).cuda()
        unet.share_cache['ref_context'] = single_null_context.repeat(
            len(ref_z0), 1, 1)
        unet.share_cache['masks'] = empty_mask
        unet.share_cache['classifier'] = face_guidence
        unet.share_cache['step_to_launch_face_guidence'] = step_to_launch_face_guidence
    
        diffusion.classifier = face_guidence
    
        # sample
        with amp.autocast(dtype=cfg.flash_dtype), torch.no_grad():
            z0 = diffusion.sample(solver=solver,
                                  noise=torch.empty(num_sample, 4,
                                                    SIZE // 8,
                                                    SIZE // 8,
                                                    device=gpu).normal_(),
                                  model=unet,
                                  model_kwargs=[c, nc],
                                  steps=steps,
                                  guide_scale=text_control_scale,
                                  guide_rescale=0.5,
                                  show_progress=True,
                                  discretization=cfg.discretization)
    
        imgs = autoencoder.decode(z0 / cfg.ae_scale)
    del unet.share_cache['ori_similarity']
    torch.cuda.empty_cache()
    unet.share_cache.clear()
    del unet, clip, autoencoder, diffusion
    # output
    imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(
        0, 255).astype(np.uint8)

    # convert to PIL image
    imgs = [Image.fromarray(img) for img in imgs]
    imgs = imgs + show_refs

    return imgs

import argparse
import yaml
from models import *
from torchvision import transforms
import torch
import PIL
from torchvision import utils as vutils
from experiment import VAEXperiment
import cv2
import numpy as np


def parse_arg():
    args = argparse.ArgumentParser()
    args.add_argument('--config', '-c', dest="filename", metavar='FILE',
                      help='path to the config file', default='configs/vae.yaml')
    args.add_argument('--checkpoint', '-ckpt', dest="ckpt",
                      metavar='FILE', help='path to the checkpoint file', default=None)
    args.add_argument('--sample', '-s', dest="sample", metavar='FILE',
                      help='path to the sample file', default=None)

    return args.parse_args()


def draw_mask(img, x, y, w, h):
    img = np.asarray(img)
    cv2.rectangle(img, (x, w), (y, h), (0, 0, 0), -1)
    img = PIL.Image.fromarray(img)
    return img


def fill_mask(img, res_img, x, y, w, h):
    img = np.array(img)
    res_img = np.array(res_img)
    img[min(w, h):max(w, h)+1,
        min(x, y):max(x, y)+1] = res_img[min(w, h):max(w, h)+1,
                                         min(x, y):max(x, y)+1]
    img = PIL.Image.fromarray(img)
    return img


def random_erase(img_tensor):
    tr_function = transforms.RandomErasing(
        p=1.0,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3),
        value=0
    )
    return tr_function.get_params(img_tensor, scale=(0.02, 0.33), ratio=(0.3, 3.3))


def main():
    args = parse_arg()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    model = vae_models[config['model_params']
                       ['name']](**config['model_params'])
    experiment = VAEXperiment(model,
                              config['exp_params'])
    experiment.load_from_checkpoint(args.ckpt,
                                    vae_model=model,
                                    params=config['exp_params'])

    tr = transforms.Compose([  # transforms.CenterCrop(148),
                            transforms.Resize(
                                (config['data_params']['patch_size'], 64)),
                            transforms.ToTensor(),
                            # transforms.RandomErasing(
                            #    p=1.0,
                            #    scale=(0.02, 0.33),
                            #    ratio=(0.3, 3.3),
                            #    value=0
                            # )
                            ])
    tensor_to_pil = transforms.ToPILImage()
    pil_to_tensor = transforms.ToTensor()

    img = PIL.Image.open(args.sample)
    img_tensor = tr(img)
    img_params = random_erase(img_tensor)
    img_pil = tensor_to_pil(img_tensor)
    img_pil = draw_mask(img_pil,
                        img_params[0],
                        img_params[1],
                        img_params[2],
                        img_params[3])
    img_tensor = pil_to_tensor(img_pil)

    vutils.save_image(img_tensor, args.sample[:-4]+'_masked.png')

    if img_tensor.shape[0] > 3:
        img_tensor = img_tensor[:-1]
    experiment.eval()
    with torch.no_grad():
        result_img = experiment(img_tensor.unsqueeze(0))
    vutils.save_image(result_img[0], args.sample[:-4]+'_result.png')
    print(result_img[0][0].shape)
    print(img_tensor.shape)
    inpainted_img = fill_mask(tensor_to_pil(img_tensor),
                              tensor_to_pil(result_img[0][0]),
                              img_params[0],
                              img_params[1],
                              img_params[2],
                              img_params[3])
    inpainted_img.save(args.sample[:-4]+'_inpainted.png')


if __name__ == '__main__':
    main()

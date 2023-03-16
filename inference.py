import argparse
import yaml
from models import *
from torchvision import transforms
import torch
import PIL
from torchvision import utils as vutils
from experiment import VAEXperiment


def parse_arg():
    args = argparse.ArgumentParser()
    args.add_argument('--config', '-c', dest="filename", metavar='FILE',
                      help='path to the config file', default='configs/vae.yaml')
    args.add_argument('--checkpoint', '-ckpt', dest="ckpt",
                      metavar='FILE', help='path to the checkpoint file', default=None)
    args.add_argument('--sample', '-s', dest="sample", metavar='FILE',
                      help='path to the sample file', default=None)

    return args.parse_args()


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
                            transforms.RandomErasing(
                                p=1.0,
                                scale=(0.02, 0.33),
                                ratio=(0.3, 3.3),
                                value=0
                            )])
    img = PIL.Image.open(args.sample)
    img_tensor = tr(img)
    vutils.save_image(img_tensor, args.sample[:-4]+'_masked.png')
    print(img_tensor.shape)
    if img_tensor.shape[0] > 3:
        img_tensor = img_tensor[:-1]
    experiment.eval()
    with torch.no_grad():
        result_img = experiment(img_tensor.unsqueeze(0))
    vutils.save_image(result_img[0], args.sample[:-4]+'_result.png')


if __name__ == '__main__':
    main()

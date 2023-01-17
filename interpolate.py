import sys
# from model import Net
sys.path.append('../../latent-diffusion')
# from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.models.diffusion.ddpm import LatentDiffusion, DiffusionWrapper
from ldm.models.diffusion.ddim import DDIMSampler
from main import DataModuleFromConfig
import copy
from omegaconf import OmegaConf
import argparse
import torch, torchvision

@torch.no_grad()
def sample(model,device='cuda'):
    torch.manual_seed(23)
    ddim_sampler = DDIMSampler(model)
    shape = (4,32,32)
    steps = 50
    batch_size=1
    samples, intermediates = ddim_sampler.sample(steps,batch_size,
                                                    shape=shape,cond=None,verbose=False)
    image = (model.decode_first_stage(samples)[0]+1)/2
    return image.clamp(0,1)

# interpolation before/after permutation
def interpolate(model,sd0,sd1,sd=None,device='cuda'):
    if not sd:
        sd = copy.deepcopy(sd1) # reuse container
    x_samples = []
    for s in [0.,.2,.4,.6,.8,1.]:
        for key in sd:
            sd[key] = (1 - s) * sd0[key] + s * sd1[key]
        model.load_state_dict(sd)
        model = model.to(device)
        print("s =", s)
        x_samples.append(sample(model, device))
    x_flattened = torchvision.utils.make_grid(x_samples,nrow=6)
    x_pil = torchvision.transforms.functional.to_pil_image(x_flattened)
    path = 'sample.png'
    x_pil.save(path)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interpolation')
    parser.add_argument('--seed', type=int, default=23,
                        help='Seed')
    parser.add_argument('--model0', type=str, default="cat.ckpt",
                        help='Loading the principal model')
    parser.add_argument('--model1', type=str, default="wild.ckpt",
                        help='Loading the alternative model')
    
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cfg = OmegaConf.load('configs/latent-diffusion/new-afhqcat-ldm-vq-4.yaml')
    model = LatentDiffusion(**cfg['model']['params']).eval()
    data0 = DataModuleFromConfig(**cfg['data']['params'])
    cfg1 = OmegaConf.load('configs/latent-diffusion/new-afhqwild-ldm-vq-4.yaml')
    data1 = DataModuleFromConfig(**cfg1['data']['params'])

    sd0 = torch.load(args.model0)['state_dict'] # fixed
    sd1 = torch.load(args.model1)['state_dict'] # fixed 
    sd2 = torch.load(args.model1)['state_dict'] # to be aligned from sd1 to sd0
    sd = torch.load(args.model1)['state_dict']  # container
    
    interpolate(model,sd0,sd1,sd,device=device)
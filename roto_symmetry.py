import torch
from model import Net
from data import get_training_set, get_test_set
import argparse
import torch.nn.functional as F
from torchvision import datasets, transforms
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import copy
from math import log10
import ot
    
def test(model,device,testing_data_loader):
    avg_psnr = 0
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


# interpolation before/after permutation
def interpolate(sd0,sd1,sd=None,device='cuda'):
    if not sd:
        sd = copy.deepcopy(sd1)
    for s in [0.,.2,.4,.6,.8,1.]:
        model = Net(args.upscale_factor).to(device)
        for key in sd:
            sd[key] = (1 - s) * sd0[key] + s * sd1[key]
        model.load_state_dict(sd)
        print("s =", s)
        test(model, device, test_loader)

# compute rotation P to align sd1 onto sd0
def compute_P(sd0,sd1,device = 'cuda'):
    P = OrderedDict()
    key0 = list(sd0.keys())[0]
    P[0] = torch.eye(sd0[key0].shape[1], device = device) # input channel P[0]
    for key in sd0:
        if len(sd0[key].shape) > 1: # ignore bias
        # save P for sd['conv1.weight'], sd['fc1.weight'], not for sd['conv1.bias'], sd['fc1.bias']
            P[key] = torch.eye(sd0[key].shape[0], device = device)
            key_1 = key
    keys = list(P.keys())
    print('computing P with layers:', keys)
    old_loss = - 1
    K = 20
    for k in range(K):
        new_loss = 0
        for i, key in enumerate(keys):
            if key in [0, keys[-1]]:
                continue
            prevkey = keys[i-1]
            nextkey = keys[i+1]
            if len(sd0[key].shape) > 2: # conv
                assert len(sd0[key].shape) == 4
#                 print(sd0[key][P[prevkey]].shape)
                W0 = sd0[key]
                W1 = torch.einsum('oikl,iI->oIkl',sd1[key],P[prevkey])  # aligned
                W0next = sd0[nextkey]
                W1next = torch.einsum('oikl,oO->Oikl',sd1[nextkey],P[nextkey])
                C = torch.einsum('oikl,Oikl->oO',W0,W1) \
                   +torch.einsum('oikl,oIkl->iI',W0next,W1next)
#             else: # fc
#                 assert len(sd0[key].shape) == 2
#                 C = torch.einsum('oi,Oi->oO',sd0[key][:,P[prevkey]],sd1[key]) \
#                    +torch.einsum('oi,oI->iI',sd0[nextkey][P[nextkey]],sd1[nextkey])
            C = C.cpu()
            a = b = torch.ones(P[key].shape[0])
            newP = ot.sinkhorn(a,b,C,eps)
            new_loss += sum(sum(C * P))
            P[key] = newP.to(device)
        print('|C| from', old_loss, 'to', new_loss)
        if abs(new_loss - old_loss) < 1e-6: # converges
            print('greedy algorithm terminates at round', k+1, '/', K)
            break
        old_loss = new_loss
    return P
    
# apply P to align sd1 onto sd0 (sd2 = sd1)
def apply_P(P, sd2):
    keys = list(P.keys())
    for i, key in enumerate(keys):
        if key == 0: # skip the first layer
            continue
        prevkey = keys[i-1]
        print('processing', key, '(second dim) , len(p_prev) = ', len(P[prevkey]))
        sd2[key] = torch.einsum('oikl,iI->oIkl',sd2[key],P[prevkey])
        print('processing', key, ', len(p) = ', len(P[key]))
        sd2[key] = torch.einsum('oikl,oO->Oikl',sd2[key],P[key])
        key_bias = key[:-6]+'bias'
        if key_bias in sd2:
            print('processing', key_bias, ', len(p) = ', len(P[key]))
            sd2[key_bias] = torch.einsum('o,oO->O',sd2[key_bias],P[key])
    return sd2
            
def main():
    parser = argparse.ArgumentParser(description='Roto-Equivariance')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--model0', type=str, default="model_epoch_30_seed_1.pth",
                        help='Loading the principal model')
    parser.add_argument('--model1', type=str, default="model_epoch_30_seed_2023.pth",
                        help='Loading the alternative model')
    parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    test_set = get_test_set(args.upscale_factor)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, num_workers=args.threads, batch_size=args.test_batch_size, shuffle=False)

    sd0 = torch.load(args.model0).to(device).state_dict() # fixed
    sd1 = torch.load(args.model1).to(device).state_dict() # fixed 
    sd2 = torch.load(args.model1).to(device).state_dict() # to be aligned from sd1 to sd0
    sd = torch.load(args.model1).to(device).state_dict() # reused interpolation container
    
    P = compute_P(sd0,sd1,device = device)
    sd2 = apply_P(P, sd2)
    # before alignment
    interpolate(sd0,sd1,sd,device=device)
    interpolate(sd0,sd2,sd,device=device)
    
    
    
if __name__ == '__main__':
    main()
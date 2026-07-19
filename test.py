from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
import torch.distributed as dist
import torch.utils.data as data
import torch.nn.functional as F
from models import *
from datasets import *
from common.utils import *
from utils import *
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..'))

args = get_params()
setup(args)
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

def main():
    args.local_rank = 0
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = eval(args.model.name)(**args.model.params)
    model.to(device)
    test_dataloader = get_dataloader(args, 'test')

    ckpt_load_path = args.model.resume
    print(f'ckpt_load_path: {ckpt_load_path}')
    if not ckpt_load_path:
        raise ValueError("You must load a checkpoint by specifying the `model.resume` argument.")
    checkpoint = torch.load(ckpt_load_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        sd = checkpoint['state_dict']
    else:
        sd = checkpoint
    model.load_state_dict(sd, strict=False)
    args.test.dataset.params.split = 'test'
    test(test_dataloader, model, args)

def test(dataloader, model,args):
    model.set_segment(args.test.dataset.params.num_segments)
    model.eval()
    y_outputs, y_labels = [], []
    with torch.no_grad():
        for _, datas in enumerate(tqdm(dataloader)):
            images, labels, video_paths, segment_indices,max_idxspic = datas
            images = images.cuda(args.local_rank)
            labels = labels.cuda(args.local_rank)
            max_idxspic = max_idxspic.cuda(args.local_rank)
            combined_input = torch.cat((images, max_idxspic), dim=1)
            outputs = model(combined_input)
            y_outputs.extend(outputs)
            y_labels.extend(labels)

    gather_y_outputs = gather_tensor(y_outputs, dist_=False, to_numpy=False)
    gather_y_labels  = gather_tensor(y_labels, dist_=False, to_numpy=False)
    if gather_y_labels.dim() == 2 and gather_y_labels.size(1) == 2:
        gather_y_labels = torch.argmax(gather_y_labels, dim=1)
    acc, real_acc, fake_acc, _, _ = compute_metrics(gather_y_outputs, gather_y_labels)
    _, y_outputs = torch.max(gather_y_outputs, dim=1)
    auc = roc_auc_score(gather_y_labels.cpu().numpy(), y_outputs.cpu().numpy())

    result = f'ACC: {acc:.4f} RealACC: {real_acc:.4f} FakeACC: {fake_acc:.4f}  AUC: {auc:.4f}'
    if args.local_rank == 0:
        print(result)

if __name__ == "__main__":
    main()


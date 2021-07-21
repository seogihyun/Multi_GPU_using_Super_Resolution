import argparse
import os
import copy
import time
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from models import FSRCNN_x
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr
from parallel import DataParallelCriterion
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel




def main():
    args = parser.parse_args()

    """ GPU device 설정 """
    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    
    """ 사용가능한 GPU 개수 반환 """
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    

def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()
    print("Use GPU: {} for training".format(gpu))

    """ scale별 weight 저장 경로 설정 """
    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    """ 저장 경로 없을 시 생성 """
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    torch.manual_seed(args.seed)

    args.rank = args.rank * ngpus_per_node + gpu

    """ 각 GPU마다 분산 학습을 위한 초기화 실행 """
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    """ 모델 생성 """
    print('==> 모델 만드는 중..')
    torch.cuda.set_device(args.gpu)
    model = FSRCNN_x(scale_factor=args.scale).cuda(args.gpu)
    model = DistributedDataParallel(model, device_ids=[args.gpu])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('모델의 파라미터 수 : ', num_params)


    train_dataset = TrainDataset(args.train_file)
    train_sampler = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=(train_sampler is None),
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  sampler=train_sampler)

    """ loss 및 optimizer 설정 """
    criterion = nn.MSELoss()
    criterion = DataParallelCriterion(criterion, device_ids=[args.gpu])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    """ Training 시작 """
    for epoch in range(args.num_epochs):
        model.train()
        """  epoch loss reset """ 
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=100) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:

                inputs, labels = data

                inputs = inputs.cuda(args.gpu)
                labels = labels.cuda(args.gpu)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg), GPU='{:1d}'.format(args.gpu))
                t.update(len(inputs))

        """ epoch 별 가중치를 설정한 경로에 저장 """
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'multiGpu_epoch_{}.pth'.format(epoch)))

        model.eval()

        """ epoch psnr reset """ 
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.cuda(args.gpu)
            labels = labels.cuda(args.gpu)

            # torch.no_grad() : impacts the autograd engine and deactivate it
            """ 가중치 업데이트 비활성화 """
            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--eval_file', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--gpu_devices', type=int, nargs='+', required=True)
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:1234', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--weights_file', type=str)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)

    main()
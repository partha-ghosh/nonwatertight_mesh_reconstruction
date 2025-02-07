from email import message
import os
import sys

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import torch
import torch.optim as optim
import open3d as o3d

import numpy as np; np.set_printoptions(precision=4)
import shutil, argparse, time
from torch.utils.tensorboard import SummaryWriter

from src import config
from src.data import collate_remove_none, collate_stack_together, worker_init_fn
from src.training import Trainer
from src.model import Encode2Points, UNET3D
from src.utils import load_config, initialize_logger, \
AverageMeter, load_model_manual

from torch.utils.data import Dataset, DataLoader
import os


class PCDataset(Dataset):
    def __init__(self, root_dir, cfg):
        # data loading
        self.cfg = cfg
        self.root_dir = root_dir
        for _, _, filenames in os.walk(f'{root_dir}/pointclouds'):
            self.filenames = filenames

    def __getitem__(self, index):
        pointcloud = np.load(f'{self.root_dir}/pointclouds/{self.filenames[index]}')
        psr = np.load(f'{self.root_dir}/psrs/{self.filenames[index]}')
        mask = np.load(f'{self.root_dir}/masks/{self.filenames[index]}')

        pointcloud_n = self.cfg['data']['pointcloud_n']
        pointcloud = pointcloud[np.random.randint(0, len(pointcloud), pointcloud_n)]

        pointcloud_noise = self.cfg['data']['pointcloud_noise']
        pointcloud = pointcloud + pointcloud_noise * np.random.randn(*pointcloud.shape)

        pointcloud_outlier_ratio = self.cfg['data']['pointcloud_outlier_ratio']
        n_outliers = int(len(pointcloud) * pointcloud_outlier_ratio)
        outlier_indices = np.random.randint(0, len(pointcloud), n_outliers)
        pointcloud[outlier_indices] = np.random.uniform(0, 1, (n_outliers, 3))

        return {
            'pointcloud': torch.tensor(pointcloud, dtype=torch.float),
            'psr': torch.tensor(psr, dtype=torch.float),
            'mask': torch.tensor(mask, dtype=torch.float)[None,:]}

    def __len__(self):
        return len(self.filenames)

def main():
    parser = argparse.ArgumentParser(description='MNIST toy experiment')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')    
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--unet', type=str, help='network architecture')
    
    args = parser.parse_args()
    cfg = load_config(args.config, 'configs/default.yaml')
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    input_type = cfg['data']['input_type']
    batch_size = cfg['train']['batch_size']
    model_selection_metric = cfg['train']['model_selection_metric']

    # PYTORCH VERSION > 1.0.0
    assert(float(torch.__version__.split('.')[-3]) > 0)

    # boiler-plate
    if cfg['train']['timestamp']:
        cfg['train']['out_dir'] += '_' + time.strftime("%Y_%m_%d_%H_%M_%S")
    logger = initialize_logger(cfg)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    shutil.copyfile(args.config, os.path.join(cfg['train']['out_dir'], 'config.yaml'))

    logger.info("using GPU: " + torch.cuda.get_device_name(0))

    # TensorboardX writer
    tblogdir = os.path.join(cfg['train']['out_dir'], "tensorboard_log")
    if not os.path.exists(tblogdir):
        os.makedirs(tblogdir, exist_ok=True)
    writer = SummaryWriter(log_dir=tblogdir)


    inputs = None

    train_dataset = PCDataset('dat/train', cfg)
    val_dataset = PCDataset('dat/test', cfg)
    vis_dataset = PCDataset('dat/test', cfg)

    train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=cfg['train']['n_workers'], shuffle=True,
    worker_init_fn=worker_init_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=8, num_workers=cfg['train']['n_workers_val'], shuffle=False,
    worker_init_fn=worker_init_fn)

    vis_loader = torch.utils.data.DataLoader(
        vis_dataset, batch_size=1, num_workers=cfg['train']['n_workers_val'], shuffle=True,
    worker_init_fn=worker_init_fn)
    
    exec(f'features2 = {args.unet}', globals())
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(Encode2Points(cfg)).to(device)
        unet = torch.nn.DataParallel(UNET3D(1, 1, features=features2)).to(device)
        # unet = torch.nn.DataParallel(UNET3D(1, 1, features=[8, 16, 32, 64, 128])).to(device)
    else:
        model = Encode2Points(cfg).to(device)
        unet = UNET3D(1, 1, features=features2).to(device)
        # unet = UNET3D(1, 1, features=[8, 16, 32, 64, 128]).to(device)

    n_parameter = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Number of parameters: %d'% n_parameter)
    # load model
    try:
        # load model
        state_dict = torch.load(os.path.join(cfg['train']['out_dir'], 'model_best.pt'))
        load_model_manual(state_dict['state_dict'], model)
            
        out = "Load model from iteration %d" % state_dict.get('it', 0)
        logger.info(out)
        # load point cloud

        # load unet
        unet.load_state_dict(torch.load(os.path.join(cfg['train']['out_dir'], 'unet_best.pt')))
        unet.eval()
    except:
        state_dict = dict()

    state_dict['loss_val_best'] = np.inf
    metric_val_best = state_dict.get(
    'loss_val_best', np.inf)

    logger.info('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))

    LR = float(cfg['train']['lr'])
    optimizer = optim.Adam(list(model.parameters()) + list(unet.parameters()), lr=LR)

    start_epoch = state_dict.get('epoch', -1)
    it = state_dict.get('it', -1)

    trainer = Trainer(cfg, optimizer, device=device)
    runtime = {}
    runtime['all'] = AverageMeter()
    
    # training loop
    for epoch in range(start_epoch+1, cfg['train']['total_epochs']+1):

        for bn, batch in enumerate(train_loader):
            it += 1
            
            start = time.time()
            loss, loss_each = trainer.train_step(inputs, batch, model, unet)

            # measure elapsed time
            end = time.time()
            runtime['all'].update(end - start)

            if it % cfg['train']['print_every'] == 0:
                log_text = ('[Epoch %02d] it=%d, batch_no=%d, loss=%.4f') %(epoch, it, bn, loss)
                writer.add_scalar('train/loss', loss, it)
                if loss_each is not None:
                    for k, l in loss_each.items():
                        if l.item() != 0.:
                            log_text += (' loss_%s=%.4f') % (k, l.item())
                        writer.add_scalar('train/%s' % k, l, it)
                
                log_text += (' time=%.3f / %.2f') % (runtime['all'].val, runtime['all'].sum)
                logger.info(log_text)

            if (it>0)& (it % cfg['train']['visualize_every'] == 0):
                os.system('rm -rf /home/scholar/tmp/shape_as_points/out/demo_shapenet_outlier/vis/mesh/*')
                os.system('rm -rf /home/scholar/tmp/shape_as_points/out/demo_shapenet_outlier/vis/pointcloud/*')
                
                for i, batch_vis in enumerate(vis_loader):
                    trainer.save(model, unet, batch_vis, it, i)
                    if i == 49:
                        break
                logger.info('Saved mesh and pointcloud')
                exit()

            # run validation
            if it > 0 and (it % cfg['train']['validate_every']) == 0:
                eval_dict = trainer.evaluate(val_loader, model, unet)
                metric_val = eval_dict[model_selection_metric]
                logger.info('Validation metric (%s): %.4f'
                    % (model_selection_metric, metric_val))
                
                for k, v in eval_dict.items():
                    writer.add_scalar('val/%s' % k, v, it)

                if  -(metric_val - metric_val_best) >= 0:
                    metric_val_best = metric_val
                    logger.info('New best model (loss %.4f)' % metric_val_best)
                    state = {'epoch': epoch,
                            'it': it,
                            'loss_val_best': metric_val_best}
                    state['state_dict'] = model.state_dict()
                    torch.save(state, os.path.join(cfg['train']['out_dir'], 'model_best.pt'))
                    torch.save(unet.state_dict(), os.path.join(cfg['train']['out_dir'], 'unet_best.pt'))

            # save checkpoint
            if (epoch > 0) & (it % cfg['train']['checkpoint_every'] == 0):
                state = {'epoch': epoch,
                         'it': it,
                         'loss_val_best': metric_val_best}
                pcl = None
                state['state_dict'] = model.state_dict()
                
                torch.save(state, os.path.join(cfg['train']['out_dir'], 'model.pt'))
                torch.save(unet.state_dict(), os.path.join(cfg['train']['out_dir'], 'unet.pt'))

                if (it % cfg['train']['backup_every'] == 0):
                    torch.save(state, os.path.join(cfg['train']['dir_model'], '%04d' % it + '.pt'))
                    torch.save(unet.state_dict(), os.path.join(cfg['train']['dir_model'], '%04d' % it + 'unet.pt'))
                    logger.info("Backup model at iteration %d" % it)
                logger.info("Save new model at iteration %d" % it)

            done=time.time()

if __name__ == '__main__':
    main()
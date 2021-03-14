import argparse
import logging
import torch
import time
import numpy as np
from pathlib import Path
from threading import Thread

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import test
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.loss import compute_loss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import select_device, intersect_dicts, ModelEMA

logger = logging.getLogger(__name__)


def distill(hyp, opt, device, tb_writer=None, wandb=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank
    
    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    # Teacher
    ckpt = torch.load(weights, map_location=device)
    model_teacher = Model(ckpt['model'].yaml, ch=3, nc=nc).to(device)  # create teacher
    state_dict = ckpt['model'].float().state_dict()
    state_dict = intersect_dicts(state_dict, model_teacher.state_dict()) # intersect
    model_teacher.load_state_dict(state_dict, strict=False)  # load
    logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model_teacher.state_dict()), weights))  # report
    
    print(model_teacher.model)

    # Student
    model_student = Model(opt.cfg, ch=3, nc=nc).to(device)  # create student
    print(model_student.model)

    # Distill Module
    # print(model_teacher.model)
    # print(model_student.model)

    # Optimizer
    # nbs = 64  # nominal batch size
    # accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    # hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    # logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    # pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    # for k, v in model_student.named_modules():
    #     if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
    #         pg2.append(v.bias)  # biases
    #     if isinstance(v, nn.BatchNorm2d):
    #         pg0.append(v.weight)  # no decay
    #     elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
    #         pg1.append(v.weight)  # apply decay

    # if opt.adam:
    #     optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    # else:
    #     optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    # optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    # optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    # logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    # del pg0, pg1, pg2

    # # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    # lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # # Logging
    # if wandb and wandb.run is None:
    #     opt.hyp = hyp  # add hyperparameters
    #     wandb_run = wandb.init(config=opt, resume="allow",
    #                            project='YOLOv3' if opt.project == 'runs/distill' else Path(opt.project).stem,
    #                            name=save_dir.stem,
    #                            id=ckpt.get('wandb_id') if 'ckpt' in locals() else None)
    # loggers = {'wandb': wandb}  # loggers dict

    # start_epoch, best_fitness = 0, 0.0

    # # Image sizes
    # gs = int(model_student.stride.max())  # grid size (max stride)
    # nl = model_student.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    # imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # # EMA
    # ema = ModelEMA(model_student)

    # # Trainloader
    # dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
    #                                         hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
    #                                         world_size=opt.world_size, workers=opt.workers,
    #                                         image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    # mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    # nb = len(dataloader)  # number of batches
    # assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # # Process 0
    # ema.updates = start_epoch * nb // accumulate  # set EMA updates
    # testloader = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt,  # testloader
    #                                 hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
    #                                 world_size=opt.world_size, workers=opt.workers,
    #                                 pad=0.5, prefix=colorstr('val: '))[0]

    # # Not resume
    # labels = np.concatenate(dataset.labels, 0)
    # c = torch.tensor(labels[:, 0])  # classes
    # # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
    # # model._initialize_biases(cf.to(device))

    # # Plot labels
    # if plots:
    #     plot_labels(labels, save_dir, loggers)
    #     if tb_writer:
    #         tb_writer.add_histogram('classes', c, 0)

    # # Anchors
    # if not opt.noautoanchor:
    #     check_anchors(dataset, model=model_student, thr=hyp['anchor_t'], imgsz=imgsz)

    # # Model parameters
    # hyp['box'] *= 3. / nl  # scale to layers
    # hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    # hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    # model_student.nc = nc  # attach number of classes to model
    # model_student.hyp = hyp  # attach hyperparameters to model
    # model_student.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    # model_student.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    # model_student.names = names

    # # Start training
    # t0 = time.time()
    # nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    # maps = np.zeros(nc)  # mAP per class
    # results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    # scheduler.last_epoch = start_epoch - 1  # do not move
    # scaler = amp.GradScaler(enabled=cuda)
    # logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
    #             f'Using {dataloader.num_workers} dataloader workers\n'
    #             f'Logging results to {save_dir}\n'
    #             f'Starting training for {epochs} epochs...')

    # for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
    #     model_student.train()

    #     # Update mosaic border
    #     # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
    #     # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

    #     mloss = torch.zeros(4, device=device)  # mean losses
    #     pbar = enumerate(dataloader)
    #     logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))
    #     pbar = tqdm(pbar, total=nb)  # progress bar

    #     optimizer.zero_grad()
    #     for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
    #         ni = i + nb * epoch  # number integrated batches (since train start)
    #         imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

    #         # Warmup
    #         if ni <= nw:
    #             xi = [0, nw]  # x interp
    #             # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
    #             accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
    #             for j, x in enumerate(optimizer.param_groups):
    #                 # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
    #                 x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
    #                 if 'momentum' in x:
    #                     x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

    #         # Forward
    #         with amp.autocast(enabled=cuda):
    #             pred = model_student(imgs)  # forward
    #             loss, loss_items = compute_loss(pred, targets.to(device), model_student)  # loss scaled by batch_size
    #             if opt.quad:
    #                 loss *= 4.

    #         # Backward
    #         scaler.scale(loss).backward()

    #         # Optimize
    #         if ni % accumulate == 0:
    #             scaler.step(optimizer)  # optimizer.step
    #             scaler.update()
    #             optimizer.zero_grad()
    #             if ema:
    #                 ema.update(model_student)

    #         # Print
    #         mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
    #         mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
    #         s = ('%10s' * 2 + '%10.4g' * 6) % (
    #             '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
    #         pbar.set_description(s)

    #         # Plot
    #         if plots and ni < 3:
    #             f = save_dir / f'train_batch{ni}.jpg'  # filename
    #             Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
    #             # if tb_writer:
    #             #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
    #             #     tb_writer.add_graph(model, imgs)  # add model to tensorboard
    #         elif plots and ni == 3 and wandb:
    #             wandb.log({"Mosaics": [wandb.Image(str(x), caption=x.name) for x in save_dir.glob('train*.jpg')]})

    #         # end batch ------------------------------------------------------------------------------------------------

    #     # Scheduler
    #     lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
    #     scheduler.step()

    #     # mAP
    #     if ema:
    #         ema.update_attr(model_student, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
    #     final_epoch = epoch + 1 == epochs
    #     if not opt.notest or final_epoch:  # Calculate mAP
    #         results, maps, times = test.test(opt.data,
    #                                             batch_size=total_batch_size,
    #                                             imgsz=imgsz_test,
    #                                             model=ema.ema,
    #                                             single_cls=opt.single_cls,
    #                                             dataloader=testloader,
    #                                             save_dir=save_dir,
    #                                             plots=plots and final_epoch,
    #                                             log_imgs=opt.log_imgs if wandb else 0)

    #     # Write
    #     with open(results_file, 'a') as f:
    #         f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    #     if len(opt.name) and opt.bucket:
    #         os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

    #     # Log
    #     tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
    #             'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
    #             'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
    #             'x/lr0', 'x/lr1', 'x/lr2']  # params
    #     for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
    #         if tb_writer:
    #             tb_writer.add_scalar(tag, x, epoch)  # tensorboard
    #         if wandb:
    #             wandb.log({tag: x})  # W&B

    #     # Update best mAP
    #     fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
    #     if fi > best_fitness:
    #         best_fitness = fi

    #     # Save model
    #     save = (not opt.nosave) or (final_epoch and not opt.evolve)
    #     if save:
    #         with open(results_file, 'r') as f:  # create checkpoint
    #             ckpt = {'epoch': epoch,
    #                     'best_fitness': best_fitness,
    #                     'training_results': f.read(),
    #                     'model': ema.ema,
    #                     'optimizer': None if final_epoch else optimizer.state_dict(),
    #                     'wandb_id': wandb_run.id if wandb else None}

    #         # Save last, best and delete
    #         torch.save(ckpt, last)
    #         if best_fitness == fi:
    #             torch.save(ckpt, best)
    #         del ckpt
    #     # end epoch ----------------------------------------------------------------------------------------------------
    # # end training

    # # Strip optimizers
    # final = best if best.exists() else last  # final model
    # for f in [last, best]:
    #     if f.exists():
    #         strip_optimizer(f)  # strip optimizers
    # if opt.bucket:
    #     os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload

    # # Plots
    # if plots:
    #     plot_results(save_dir=save_dir)  # save as results.png
    #     if wandb:
    #         files = ['results.png', 'precision_recall_curve.png', 'confusion_matrix.png']
    #         wandb.log({"Results": [wandb.Image(str(save_dir / f), caption=f) for f in files
    #                                 if (save_dir / f).exists()]})
    #         if opt.log_artifacts:
    #             wandb.log_artifact(artifact_or_path=str(final), type='model', name=save_dir.stem)

    # # Test best.pt
    # logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    # if opt.data.endswith('coco.yaml') and nc == 80:  # if COCO
    #     for conf, iou, save_json in ([0.25, 0.45, False], [0.001, 0.65, True]):  # speed, mAP tests
    #         results, _, _ = test.test(opt.data,
    #                                     batch_size=total_batch_size,
    #                                     imgsz=imgsz_test,
    #                                     conf_thres=conf,
    #                                     iou_thres=iou,
    #                                     model=attempt_load(final, device).half(),
    #                                     single_cls=opt.single_cls,
    #                                     dataloader=testloader,
    #                                     save_dir=save_dir,
    #                                     save_json=save_json,
    #                                     plots=False)

    # wandb.run.finish() if wandb and wandb.run else None
    # torch.cuda.empty_cache()
    # return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov3.pt', help='initial weights path for teacher network')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path for student network')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.distill.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
    parser.add_argument('--log-artifacts', action='store_true', help='log artifacts, i.e. final trained model')
    parser.add_argument('--workers', type=int, default=3, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/distill', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    opt = parser.parse_args()

    opt.world_size = 1
    opt.global_rank = -1
    set_logging()

    # Setup path
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.weights), '--weights must be specified for teacher network'
    assert len(opt.cfg), '--cfg must be specified for student network'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    # Device
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Distill and train
    import wandb
    logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/')
    tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
    distill(hyp, opt, device, tb_writer, wandb)

    
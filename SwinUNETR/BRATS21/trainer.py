# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pdb
import shutil
import time
import math

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch

def unpack_np(target, out_channels):
    np_target = target.numpy().astype(np.uint32)
    output = []
    for i in np.arange(0, 32, 8, dtype=np.uint32):
        np_target1 = np.right_shift(np_target, i)
        np_target2 = np.bitwise_and(np_target1, 0xFF).astype(np.uint8)
        x_bits = np.unpackbits(np_target2, axis=1).reshape(-1, 8, target.shape[2], target.shape[3], target.shape[4])
        output.append(x_bits)

    output = np.concatenate(output, axis=1) 
    output = output[:, :out_channels, :, :, :]
    target_bits = torch.from_numpy(output).float()
    return target_bits

def unpack(target, mask, shift):
    # target = torch.from_numpy(target_bits).float()
    target = torch.transpose(target, 1, -1)
    x_bits = (target.to(torch.int) & mask) >> shift
    x_bits = torch.transpose(x_bits, 1, -1)
    # x_bits = x_bits.float()

    # for s in [0, 8, 16, 32]:
    #     target1 = torch.bitwise_right_shift(target.to(torch.int), torch.tensor([s]*math.prod(target.shape)).reshape(target.shape))
    return x_bits

def pack(x_bit):
    x_bit = x_bit.to(torch.int32)
    x =x_bit[:, 0, :, :, :]
    for i in range(1, x_bit.shape[1]):
        x = torch.bitwise_or(x, x_bit[:, i, :, :, :])
    x = x.reshape((x.shape[0], 1, *x.shape[1:]))
    return x

def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        mask = torch.tensor(list([2**i for i in range(args.out_channels)]), dtype=torch.int)
        mask = mask.cuda(args.rank)
        shift = torch.arange(0, args.out_channels)
        shift = shift.cuda(args.rank)
        target = unpack(target, mask, shift)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_sigmoid=None, post_pred=None):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"], batch_data["label"]
            data = data.cuda(args.rank)
            with autocast(enabled=args.amp):
                logits = model_inferer(data)

            logits = logits.cpu()
            # mask = torch.tensor(list([2**i for i in range(args.out_channels)]), dtype=torch.int)
            # mask = mask.cuda(args.rank)
            # shift = torch.arange(0, args.out_channels)
            # shift = shift.cuda(args.rank)
            # target = target.cuda(args.rank)
            # target = unpack(target, mask, shift)
            target = unpack_np(target, args.out_channels)
            target = torch.Tensor(target)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            val_outputs_list = None
            logits = None
            target = None
            # val_labels_list = [d.cpu() for d in val_labels_list]
            # val_output_convert = [d.cpu() for d in val_output_convert]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)
            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)
            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                Dice_TC = run_acc.avg[0]
                Dice_WT = run_acc.avg[1]
                Dice_ET = run_acc.avg[2]
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    ", Dice_TC:",
                    Dice_TC,
                    ", Dice_WT:",
                    Dice_WT,
                    ", Dice_ET:",
                    Dice_ET,
                    ", time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()

    return run_acc.avg


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
    semantic_classes=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )

            if args.rank == 0:
                Dice_TC = val_acc[0]
                Dice_WT = val_acc[1]
                Dice_ET = val_acc[2]
                print(
                    "Final validation stats {}/{}".format(epoch, args.max_epochs - 1),
                    ", Dice_TC:",
                    Dice_TC,
                    ", Dice_WT:",
                    Dice_WT,
                    ", Dice_ET:",
                    Dice_ET,
                    ", time {:.2f}s".format(time.time() - epoch_time),
                )

                if writer is not None:
                    writer.add_scalar("Mean_Val_Dice", np.mean(val_acc), epoch)
                    if semantic_classes is not None:
                        for val_channel_ind in range(len(semantic_classes)):
                            if val_channel_ind < val_acc.size:
                                writer.add_scalar(semantic_classes[val_channel_ind], val_acc[val_channel_ind], epoch)
                val_avg_acc = np.mean(val_acc)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max

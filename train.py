from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

import argparse
import logging
import os
from datetime import datetime

import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from models.resfcn256 import ResFCN256
from utils.data import UVmap2Mesh
from utils.dataset import FaceDataset
from utils.visualize import logTrainingSamples

from clearml import Task
task = Task.init(project_name="Facial-landmark", task_name="PRnet-train-300WLP-val-AFLW2000")

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

BEST_LOSS = 10000.
EPOCH = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def train(args):
    global EPOCH
    model = ResFCN256()

    train_set = FaceDataset(root=args.train_root)

    test_set = FaceDataset(root=args.test_root, aug=False)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=True)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=args.test_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True)

    optimizer = Adam(params=model.parameters(), lr=1e-4)
    
    if args.pretrained:
        weight = torch.load(args.pretrained)
        model.load_state_dict(weight)

    model.to(DEVICE)
    model.train()

    # while True:
    #     test(model, test_loader, args.save_path)

    ############################# training #############################
    for epoch in range(10000):
        EPOCH = epoch
        logger.info("=> Training phase.")
        for idx, item in enumerate(train_loader):
            imgs, gtposes, _ = item
            imgs = imgs.to(DEVICE)
            gtposes = gtposes.to(DEVICE)

            optimizer.zero_grad()
            losses, metrics, _ = model(imgs, gtposes)

            loss = torch.mean(losses)
            metric = torch.mean(metrics)

            loss.backward()
            optimizer.step()

            if idx%100==99:
                writer.add_scalar('Train/Foreface-Weighted-Root-Square-Error', loss.item(), idx+epoch*len(train_loader))
                writer.add_scalar('Train/Normalized-Mean-Square-Error', metric.item(), idx+epoch*len(train_loader))
                logger.info(f"==> Epoch {epoch} - Current FWRSE: {loss.item()} - Current NME: {metric.item()}")

        ############################# testing #############################
        test(model, test_loader, args.save_path)


def test(model, test_loader, save_path):
    global BEST_LOSS
    today = datetime.today().strftime('%Y-%m-%d')
    _save_path = os.path.join(save_path, today)
    os.makedirs(_save_path, exist_ok=True)

    model.eval()
    logger.info("=> Testing phase.")
    test_loss = 0.
    with torch.no_grad():
        for idx, item in enumerate(test_loader):
            imgs, gtposes, metas = item
            imgs = imgs.to(DEVICE)
            gtposes = gtposes.to(DEVICE)

            _, metrics, poses = model(imgs, gtposes)
            test_loss += torch.mean(metrics)

            if idx==0:
                logTrainingSamples(gtposes, poses, metas, EPOCH, writer)

        test_loss /= len(test_loader)
        writer.add_scalar("Test/Normalized-Mean-Square-Error", test_loss, EPOCH)   

        if test_loss <= BEST_LOSS:
            BEST_LOSS = test_loss
            logger.info(f"==> New bess loss: {test_loss}. Saving best model...")
            torch.save(model.state_dict(), os.path.join(_save_path,"best.pth"))
            logger.info(f"==> Done saving!")
        else:
            logging.info(f"==> Current loss {test_loss} - Best loss {BEST_LOSS}. Saving last model...")    
            torch.save(model.state_dict(), os.path.join(_save_path,"last.pth"))
            logger.info(f"==> Done saving!")


def opt():
    parser = argparse.ArgumentParser(description='model arguments')
    parser.add_argument('--train-root', type=str, required=True, help='path to train data set')
    parser.add_argument('--test-root', type=str, required=True, help='path to test data set')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--test-size', type=int, default=64, help='test size')
    parser.add_argument('--save-path', type=str, required=True, help='model save path')
    parser.add_argument('--num-workers', type=int, default=8, help='number of workers to push data')
    parser.add_argument('--pretrained', type=str, help='path to pretrained weight')
    
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = opt()
    train(args)

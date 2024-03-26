import os
import time
import argparse
import time
import random
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score

from tools.log_util import create_logger
from models.basemodel import BaseModel
from dataloader.datasets import base_dataset

def seed_torch(seed=21):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    seed_torch(44)
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.rank = 0
    args.output = None
    output_dir = './output/train'
    if args.rank == 0 and output_dir == None:
        exp_name = "-".join(
            [
                time.strftime('%m-%d %H:%M:%S', time.localtime()),
            ]
        )
        output_dir = args.output if args.output else "./output/train"
        # output_dir = os.path.join(output_dir, exp_name) 
        output_dir = output_dir +'/'+ exp_name

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    _logger = create_logger(os.path.join(output_dir, 'output.log'), rank=args.rank)
    _logger.info('**********************Start logging**********************')

    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    _logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    

    model = BaseModel(embedding_dim=512, num_classes=9)
    
    _logger.info(
        f"Model created, param count:{sum([m.numel() for m in model.parameters()])}"
    )

    initepoch = 0

    model.cuda()
    _logger.info("Using single GPU.") 

    # opti & sche
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    # loss
    loss_func = torch.nn.CrossEntropyLoss()

    train_datasets, val_datasets, test_datasets = base_dataset('data/WOZ/train.tsv'), base_dataset('data/WOZ/val.tsv'), base_dataset('data/WOZ/test.tsv')
    train_dataloader = DataLoader(train_datasets,sampler = RandomSampler(train_datasets),batch_size = 256)
    val_dataloader = DataLoader(val_datasets, batch_size = 256)
    test_dataloader = DataLoader(test_datasets,batch_size = 128)
    # training

    _logger.info('**********************Start Training **********************')

    for epoch in range(initepoch, 30):
        loss_list = []
        model.train()
        time.sleep(10)
        _logger.info("Training epoch :{}".format(epoch))

        for i_iter, batch in enumerate(tqdm(train_dataloader)):  
            sentence, token_type_ids, label = batch[0].cuda(),batch[1].cuda(),batch[2].cuda()
            # forward + backward + optimize
            outputs = model(sentence, token_type_ids)
                
            loss = loss_func(outputs, label.long())
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            optimizer.zero_grad()
            if i_iter % 40 == 0:
                _logger.info(f'epoch: {epoch}/{40}, cur_iter={i_iter}/{len(train_dataloader)}, '
                            f'loss: {round(np.mean(loss_list), 4)}')
        
        checkpoint_dict = {'epoch': epoch, 
            'model_state_dict': model.state_dict(), 
            'optim_state_dict': optimizer.state_dict(),}
        if args.rank == 0: 
            torch.save(checkpoint_dict, os.path.join(output_dir,'last.pth.tar'))

        total_eval_loss = 0
        y_true = []
        y_pred = []
        model.eval()
        with torch.no_grad():
            for i_iter, batch in enumerate(tqdm(val_dataloader)):  
                sentence, token_type_ids, label = batch[0].cuda(),batch[1].cuda(),batch[2].cuda()
            
                # forward + backward + optimize
                outputs = model(sentence, token_type_ids)
                loss = loss_func(outputs, label.long())

                total_eval_loss += loss.item()        
                outputs = outputs.argmax(axis=-1).detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                y_true.extend(label.tolist())
                y_pred.extend(outputs.tolist())

            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_micro = f1_score(y_true, y_pred, average='micro')
            _logger.info(f'f1_macro:{f1_macro}, f1_micro:{f1_micro}, loss:{round(total_eval_loss/len(val_datasets), 4)}')


if __name__ == '__main__':
    main()




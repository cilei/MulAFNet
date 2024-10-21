import os, argparse, time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader

from utils import *
from loader import PretrainDataset
from models.seqTranformer import TrfmSeq2seq
from Smiles2token import Token2Idx

def parse_args():
    parser = argparse.ArgumentParser()
    # environment arguments
    parser.add_argument('--seed', default=0, type=int, help="set seed")
    parser.add_argument('--cpu', default=False, action='store_true', help="train on cpu")
    parser.add_argument('--gpu', default=0, type=int, help="gpu id")
    # directory arguments
    parser.add_argument('--data_dir', default='data/ZINC15', type=str, help="directory of pre-training data")
    parser.add_argument('--save_root', default='pretrained_model_cl_zinc15_250k/', type=str, help="root directory to save pretrained model")
    # network arguments
    parser.add_argument('--gnn', default='GIN', type=str, help="GNN architecture")
    parser.add_argument('--num_atom_layers', default=3, type=int, help="num of atom-level gnn layers")
    parser.add_argument('--num_fg_layers', default=2, type=int, help="num of FG-level gnn layers")
    parser.add_argument('--emb_dim', default=128, type=int, help="embedding dimension")
    parser.add_argument('--atom2fg_reduce', default='mean', type=str, help="atom-to-fg message passing method")
    parser.add_argument('--pool', default='mean', type=str, help="graph readout layer")
    parser.add_argument('--dropout', default=0, type=float, help="dropout rate")
    # train arguments
    parser.add_argument('--batch_size', default=512, type=int, help="batch size")
    parser.add_argument('--lr', default=1e-3, type=float, help="learning rate")
    parser.add_argument('--num_epochs', default=100, type=int, help="number of training epoch")

    args = parser.parse_args()
    return args


def main(args,start_time,model_save_dir):
    set_seed(args.seed)
    os.makedirs(model_save_dir, exist_ok=True)
    logger = create_file_logger(os.path.join(model_save_dir, 'log.txt'))
    logger.info(f"\n\n======={time.strftime('%Y-%m-%d %H:%M:%S')}=======\n")
    logger.info("=======Setting=======")
    for k in args.__dict__:
        v = args.__dict__[k]
        logger.info(f"{k}: {v}")
    
    # load data
    if not os.path.exists(args.data_dir):
        print("Data directory not found!")
        return

    train_set = PretrainDataset(root=args.data_dir,
                                mol_filename='zinc15_250k.txt',
                                fg_corpus_filename='fg_corpus.txt',
                                mol2fgs_filename='mol2fgs_list.json')
    
    logger.info(f"train data num: {len(train_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, follow_batch=['fg_x'])

    logger.info("\n=======Pre-train Start=======")
    device = torch.device('cpu' if args.cpu else ('cuda:' + str(args.gpu)))
    logger.info(f"Utilized device as {device}")
    os.chdir(model_save_dir)

    start_epoch = 0
    loss_record = []  # train loss

    model = TrfmSeq2seq(len(Token2Idx), 256, len(Token2Idx), 4).cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    for e in range(50):
        for i, mol in enumerate(train_loader):
            mask = torch.t(mol.masked_x.cuda())
            sm = torch.t(mol.smiTokenizer.cuda())
            optimizer.zero_grad()
            output = model(mask)
            loss = F.nll_loss(output.view(-1, len(Token2Idx)),
                              sm.contiguous().view(-1), ignore_index=Token2Idx['PAD'])
            loss.backward()
            optimizer.step()
            if i % 400 == 0 and i != 0:
                print(f"epoch {e}:")
                print(f"total loss: {loss}")
                if not os.path.isdir("save"):
                    os.makedirs("save")
                torch.save(model.state_dict(), './save/trfm_new_ds_lr5_%d_%d.pth' % (e, i))
        

    


if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()
    model_save_dir = os.path.join(args.save_root,
                              args.gnn +
                              '_dim'+str(args.emb_dim) +
                              '_lr'+str(args.lr))
    main(args,start_time,model_save_dir)
    
    
    
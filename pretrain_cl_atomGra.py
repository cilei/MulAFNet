import os, argparse, time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader

from utils import *
from chem import *
from loader import PretrainDataset
from models.seqTranformer import TrfmSeq2seq
from Smiles2token import Token2Idx
from models.series_gin_edge import AtomGraphGINE
from models.mol_predictor import Atom_Model_decoder

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
    parser.add_argument('--num_epochs', default=50, type=int, help="number of training epoch")
    parser.add_argument('--output_model_file', type=str, default='./save/pretrain_atom.pth',
                        help='filename to output the pre-trained model')

    args = parser.parse_args()
    return args


def train(model_decoder, train_loader, optimizer2, device):
    model_decoder.train()
    log_loss = 0
    for i, mol in enumerate(train_loader):
        mol = mol.to(device)

        optimizer2.zero_grad()
        loss = model_decoder(mol)
        loss.backward()
        optimizer2.step()
        log_loss += loss.item()
        
    print("log_loss:",log_loss)


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


    encoder2 = AtomGraphGINE(num_atom_layers=args.num_atom_layers,latent_dim=args.emb_dim,
                          atom_dim=ATOM_DIM, bond_dim=BOND_DIM,
                          dropout=args.dropout)  
    model_decoder = Atom_Model_decoder(encoder2, args.emb_dim, device)

    model_decoder.to(device)

    optimizer2 = optim.Adam([{'params': model_decoder.encoder1.parameters(), 'lr': args.lr},
                             {'params': model_decoder.atom_num_s.parameters(), 'lr': args.lr},
                            {'params': model_decoder.bond_num_s.parameters()}], lr=args.lr)
    
    for epoch in range(1, args.num_epochs + 1):
        print('====epoch',epoch)
        train(model_decoder, train_loader, optimizer2, device)
    
    if not args.output_model_file == "":
        torch.save(model_decoder.encoder1.state_dict(), args.output_model_file)


if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()
    model_save_dir = os.path.join(args.save_root,
                              args.gnn +
                              '_dim'+str(args.emb_dim) +
                              '_lr'+str(args.lr))
    main(args,start_time,model_save_dir)
    
    
    
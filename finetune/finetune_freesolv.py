import os, sys, argparse, time
import json
from tqdm import tqdm
from rdkit import Chem
import deepchem as dc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from sklearn.metrics import mean_squared_error
from SmilesEnumerator import SmilesEnumerator
from Smiles2token import smi_tokenizer
sys.path.append('..')
from utils import *
from chem import *
from loader import MoleculeNetDataset
from models.series_gin_edge import SerGINE,AtomGraphGINE,SubGraphGINE
from models.mol_predictor import MolPredictor,Classification,MolPredictorAllM
from models.seqTranformer import TrfmSeq2seq
from Smiles2token import Token2Idx,split,get_array


def testReg(model, data_loader, device):
    model.eval()
   
    with torch.no_grad():
        avg_loss = 0
        true_label, pred_score = [], []
        for mol in data_loader:
            mol = mol.to(device)

            output = model(mol)

            
            loss = torch.sum((output-mol.y.reshape(-1, args.num_tasks))**2)/output.size(0)
            loss = torch.mean(loss * mol.w.reshape(-1, args.num_tasks))
            avg_loss += loss.item()

            # output = sigmoid(output)
            for true, pred in zip(mol.y, output):
                true_label.append(true.item())
                pred_score.append(pred.item())

        avg_loss = avg_loss/len(data_loader)
        mse = mean_squared_error(true_label, pred_score)
        rmse=np.sqrt(mean_squared_error(true_label,pred_score))

    return avg_loss, rmse


def trainReg(modelAll, data_loader, optimizer, device):
    modelAll.train()
   
    log_loss = 0
    for i, mol in enumerate(data_loader):
        mol = mol.to(device)
        # mol.seqEncoderTensor.shape 1024
        optimizer.zero_grad()
        output = modelAll(mol)

        loss = torch.sum((output-mol.y.reshape(-1, args.num_tasks))**2)/output.size(0)

        # loss = criterion(output, mol.y.reshape(-1, args.num_tasks))
        loss = torch.mean(loss * mol.w.reshape(-1, args.num_tasks))
        loss.backward()
        optimizer.step()
        log_loss += loss.item()


        # log
        if (i+1) % args.log_interval == 0:
            log_loss = log_loss/args.log_interval
            print(f"batch: {i+1}/{len(data_loader)} | loss: {log_loss :.8f} | time: {time.time()-start_time :.4f}")
            log_loss = 0  


def trainAll(modelAll, data_loader, optimizer, device):
    modelAll.train()
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    log_loss = 0
    for i, mol in enumerate(data_loader):
        mol = mol.to(device)
        # mol.seqEncoderTensor.shape 1024
        optimizer.zero_grad()
        output = modelAll(mol)
        loss = criterion(output, mol.y.reshape(-1, args.num_tasks))
        loss = torch.mean(loss * mol.w.reshape(-1, args.num_tasks))
        loss.backward()
        optimizer.step()
        log_loss += loss.item()


        # log
        if (i+1) % args.log_interval == 0:
            log_loss = log_loss/args.log_interval
            print(f"batch: {i+1}/{len(data_loader)} | loss: {log_loss :.8f} | time: {time.time()-start_time :.4f}")
            log_loss = 0   

def testAll(model, data_loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    with torch.no_grad():
        avg_loss = 0
        true_label, pred_score = [], []
        for mol in data_loader:
            mol = mol.to(device)

            output = model(mol)
            loss = criterion(output, mol.y.reshape(-1, args.num_tasks))
            loss = torch.mean(loss * mol.w.reshape(-1, args.num_tasks))
            avg_loss += loss.item()

            # output = sigmoid(output)
            for true, pred in zip(mol.y, output):
                true_label.append(true.item())
                pred_score.append(pred.item())

        avg_loss = avg_loss/len(data_loader)
        auc_roc = roc_auc_score(true_label, pred_score)

    return avg_loss, auc_roc

def parse_args():
    parser = argparse.ArgumentParser()
    # environment arguments
    parser.add_argument('--cpu', default=False, action='store_true', help="train on cpu")
    parser.add_argument('--gpu', default=0, type=int, help="gpu id")
    # directory arguments
    parser.add_argument('--output_dir', default='result/Freesolv', type=str, help="output directory of task")
    parser.add_argument('--model_name', default='roc_best_model.pth', type=str, help="saved model name")
    # parser.add_argument('--time', default=1, type=int, help="time of experiment")
    # network arguments
    parser.add_argument('--gnn', default='GIN', type=str, help="GNN architecture")
    parser.add_argument('--num_atom_layers', default=3, type=int, help="num of atom-level gnn layers")
    parser.add_argument('--num_fg_layers', default=2, type=int, help="num of FG-level gnn layers")
    parser.add_argument('--emb_dim', default=128, type=int, help="embedding dimension")
    parser.add_argument('--num_tasks', default=1, type=int, help="number of tasks")
    parser.add_argument('--dropout', default=0.5, type=float, help="dropout rate")
    # training arguments
    parser.add_argument('--from_scratch', default=False, action='store_true', help="train from scratch")
    parser.add_argument('--pretrain_dir', default='../pretrained_model_cl_zinc15_250k', type=str, help="directory of pretrained models")
    parser.add_argument('--pretrain_model_name', default='model.pth', type=str, help="pretrained model name")
    # parser.add_argument('--metric', default='CosineSimilarity', type=str, help="criterion of embedding distance")
    # parser.add_argument('--margin', default=1.0, type=float, help="margin of contrastive loss")
    # parser.add_argument('--pre_lr', default=1e-3, type=float, help="learning rate of pretraining")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--lr0', default=1e-4, type=float, help="learning rate of encoder0")
    parser.add_argument('--lr1', default=1e-4, type=float, help="learning rate of encoder1")
    parser.add_argument('--lr2', default=1e-4, type=float, help="learning rate of predictor")
    parser.add_argument('--num_epochs', default=100, type=int, help="number of training epoch")
    parser.add_argument('--log_interval', default=20, type=int, help="log interval (batch/log)")
    parser.add_argument('--early_stop', default=False, action='store_true', help="use early stop strategy")
    parser.add_argument('--patience', default=20, type=int, help="num of waiting epoch")
    parser.add_argument('--weight_decay', default=0, type=float, help="weight decay")
    parser.add_argument('--splitter', default='scaffold', choices=['scaffold', 'random'], help="Split method of dataset")

    args = parser.parse_args()

    return args

def process_data(args):
    tasks, datasets, transformer = dc.molnet.load_freesolv(data_dir='./data/MoleculeNet', save_dir='./data/MoleculeNet',
                                                       splitter=args.splitter)
    dataset = [[], [], []]
    err_cnt = 0

    sme = SmilesEnumerator()
    trfm = TrfmSeq2seq(len(Token2Idx), 256, len(Token2Idx), 4)
    trfm.load_state_dict(torch.load('../save/trfm_new_ds_lr5_49_400.pth'))
    trfm.eval()

    for i in range(3):
        for X, y, w, ids in datasets[i].itersamples():
            mol = Chem.MolFromSmiles(ids)
            if mol is None:
                print(f"'{ids}' cannot be convert to graph")
                err_cnt += 1
                continue
            atom_features, bond_list, bond_features, fg_features, fg_edge_list, fg_edge_features, atom2fg_list = mol_to_graphs(mol)
            if fg_features == []:  # C
                err_cnt += 1
                print(f"{ids} cannot be converted to FG graph")
                continue
            x_split = [split(ids.strip())]
            xid, xseg = get_array(x_split)
            seqEncoderTensor = trfm.encode(torch.t(xid))
            dataset[i].append([atom_features, bond_list, bond_features, fg_features, fg_edge_list, fg_edge_features, atom2fg_list, y, w,seqEncoderTensor])
    print(f"{err_cnt} data can't be convert to graph")
    train_set, valid_set, test_set = dataset
    return train_set, valid_set, test_set


def main(args,start_time,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = create_file_logger(os.path.join(output_dir, 'log.txt'))
    logger.info("=======Setting=======")
    for k in args.__dict__:
        v = args.__dict__[k]
        logger.info(f"{k}: {v}")
    device = torch.device('cpu' if args.cpu else ('cuda:' + str(args.gpu)))
    logger.info(f"\nUtilized device as {device}")

    # load data
    logger.info("\n=======Process Data=======")

    train_set, valid_set, test_set = process_data(args)
    logger.info(f"train data num: {len(train_set)} | valid data num: {len(valid_set)} | test data num: {len(test_set)}")
    train_set = MoleculeNetDataset(train_set)
    valid_set = MoleculeNetDataset(valid_set)
    test_set = MoleculeNetDataset(test_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, follow_batch=['fg_x'])
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, follow_batch=['fg_x'])
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, follow_batch=['fg_x'])



    
    encoder1 = SubGraphGINE(num_fg_layers=args.num_fg_layers, latent_dim=args.emb_dim,
                          fg_dim=FG_DIM,fg_edge_dim=FG_EDGE_DIM,
                          dropout=args.dropout)
    
    encoder2 = AtomGraphGINE(num_atom_layers=args.num_atom_layers,latent_dim=args.emb_dim,
                          atom_dim=ATOM_DIM, bond_dim=BOND_DIM,
                          dropout=args.dropout)
    
    modelAll = MolPredictorAllM(encoder1=encoder1,encoder2=encoder2, latent_dim=1024+args.emb_dim*2,
                                num_tasks=args.num_tasks, dropout=args.dropout)
    
    modelAll.to(device)

    optimizer2 = optim.Adam([{'params': modelAll.encoder1.parameters(), 'lr': args.lr0},
                             {'params': modelAll.encoder2.parameters(), 'lr': args.lr1},
                            {'params': modelAll.predictor.parameters()}], lr=args.lr2, weight_decay=args.weight_decay)

    
    modelAll.from_pretrained_encoder2('./save/pretrain_atom.pth',device)
    modelAll.from_pretrained_encoder1('./save/pretrain_sub.pth',device)   


    logger.info("\n=======Train=======")
    os.chdir(output_dir)

    
    best = [0, 0, 1]  # [epoch, valid_roc, test_roc]
    early_stop_cnt = 0
    record = [[], [], [], [], [], []]  # [train loss, train roc, valid loss, valid roc, test loss, test roc]

    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch+1 :03d}")
        early_stop_cnt += 1
        trainReg(modelAll,train_loader,optimizer2,device)
        # train(model, train_loader, optimizer, device)
        # record
        train_loss, train_rmse = 0, 0
        train_loss, train_rmse = testReg(modelAll, train_loader, device)
        record[0].append(train_loss)
        record[1].append(train_rmse)
        valid_loss, valid_rmse = testReg(modelAll, valid_loader, device)
        record[2].append(valid_loss)
        record[3].append(valid_rmse)
        test_loss, test_rmse = testReg(modelAll, test_loader, device)
        record[4].append(test_loss)
        record[5].append(test_rmse)
        logger.info(f"Train loss: {train_loss :.8f} | RMSE: {train_rmse :.8f}")
        logger.info(f"Valid loss: {valid_loss :.8f} | RMSE: {valid_rmse :.8f}")
        logger.info(f"Test  loss: {test_loss :.8f} | RMSE: {test_rmse :.8f}")
        # update model
        if test_rmse < best[2]:
            best = [epoch+1, valid_rmse, test_rmse]
            torch.save(modelAll.state_dict(), args.model_name)
            print(f"Saved model of Epoch {epoch+1 :03d} into '{args.model_name}'")
            early_stop_cnt = 0
        else:
            print(f"No improvement since Epoch {best[0] :03d} with Valid RMSE: {best[1] :.8f} | Test RMSE: {best[2] :.8f}")
        # early stop
        if args.early_stop and (early_stop_cnt == args.patience):
            logger.info(f"Early stop at Epoch {epoch+1 :03d}")
            break
    
    logger.info(f"\n'{args.model_name}' | Epoch: {best[0] :03d} | RMSE: {best[1] :.8f}")

    logger.info("\n=======Test=======")
    logger.info(f"{args.model_name}")
    modelAll.load_state_dict(torch.load(args.model_name, map_location=device))
    logger.info("Train set:")
    _, rmse = testReg(modelAll, train_loader, device)
    logger.info(f"RMSE: {rmse :.8f}")
    logger.info("Valid set:")
    _, rmse = testReg(modelAll, valid_loader, device)
    logger.info(f"RMSE: {rmse :.8f}")
    logger.info("Test set:")
    _, rmse = testReg(modelAll, test_loader, device)
    logger.info(f"RMSE: {rmse :.8f}")

    logger.info("\n=======Finish=======")    



if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()

    output_dir = args.gnn+'_dim'+str(args.emb_dim)
    output_dir = os.path.join(args.output_dir, output_dir,
                          'lr0_'+str(args.lr0)+'_lr1_'+str(args.lr1) + '_lr2_'+str(args.lr2) + '_dropout'+str(args.dropout))
    main(args,start_time,output_dir)
    
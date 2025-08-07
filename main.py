import os
import copy
from tqdm import tqdm
from matplotlib.pyplot import axis
import torch
import random
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import argparse
from data import MovieLens1MColdStartDataLoader, TaobaoADColdStartDataLoader
from model import FactorizationMachineModel, WideAndDeep, DeepFactorizationMachineModel, AdaptiveFactorizationNetwork, ProductNeuralNetworkModel
from model import AttentionalFactorizationMachineModel, DeepCrossNetworkModel, MWUF, MetaE, CREU
from model.wd import WideAndDeep

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_model_path', default='./chkpt/')
    parser.add_argument('--dataset_name', default='movielens1M', help='required to be one of [movielens1M, taobaoAD]')
    parser.add_argument('--datahub_path', default='./datahub/')
    parser.add_argument('--warmup_model', default='creu', help="required to be one of [base, mwuf, metaE, creu]")
    parser.add_argument('--is_dropoutnet', type=bool, default=False, help="whether to use dropout net for pretrain")
    parser.add_argument('--bsz', type=int, default=16000)
    parser.add_argument('--shuffle', type=int, default=1)
    parser.add_argument('--model_name', default='deepfm', help='backbone name, we implemented [fm, wd, deepfm, afn, ipnn, opnn, afm, dcn]')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--creu_epochs', type=int, default=10)
    parser.add_argument('--creu_iters', type=int, default=10)
    parser.add_argument('--num_components', type=int, default=3, help='number of components in ensemble for CREU')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    parser.add_argument('--runs', type=int, default=5, help = 'number of executions to compute the average metrics')
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()
    return args

def get_loaders(name, datahub_path, device, bsz, shuffle):
    path = os.path.join(datahub_path, name, "{}_data.pkl".format(name))
    if name == 'movielens1M':
        dataloaders = MovieLens1MColdStartDataLoader(name, path, device, bsz=bsz, shuffle=shuffle)
    elif name == 'taobaoAD':
        dataloaders = TaobaoADColdStartDataLoader(name, path, device, bsz=bsz, shuffle=shuffle)
    else:
        raise ValueError('unkown dataset name: {}'.format(name))
    return dataloaders

def get_model(name, dl):
    if name == 'fm':
        return FactorizationMachineModel(dl.description, 16)
    elif name == 'wd':
        return WideAndDeep(dl.description, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'deepfm':
        return DeepFactorizationMachineModel(dl.description, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'afn':
        return AdaptiveFactorizationNetwork(dl.description, embed_dim=16, LNN_dim=1500, mlp_dims=(400, 400, 400), dropout=0)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(dl.description, embed_dim=16, mlp_dims=(16, ), dropout=0, method='inner')
    elif name == 'opnn':
        return ProductNeuralNetworkModel(dl.description, embed_dim=16, mlp_dims=(16, ), dropout=0, method='outer')
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(dl.description, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))
    elif name == 'dcn':
        return DeepCrossNetworkModel(dl.description, embed_dim=16, num_layers=3, mlp_dims=[16, 16], dropout=0.2)
    return

def test(model, data_loader, device):
    model.eval()
    labels, scores, predicts = list(), list(), list()
    criterion = torch.nn.BCELoss()
    with torch.no_grad():
        for _, (features, label) in enumerate(data_loader):
            features = {key: value.to(device) for key, value in features.items()}
            label = label.to(device)
            y = model(features)
            # print(y.shape)
            labels.extend(label.tolist())
            scores.extend(y.tolist())
    scores_arr = np.array(scores)
    return roc_auc_score(labels, scores), f1_score(labels, (scores_arr > np.mean(scores_arr)).astype(np.float32).tolist())

def dropoutNet_train(model, data_loader, device, epoch, lr, weight_decay, save_path, log_interval=10, val_data_loader=None):
    # train
    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), \
                                            lr=lr, weight_decay=weight_decay)
    for epoch_i in range(1, epoch + 1):
        epoch_loss = 0.0
        total_loss = 0
        total_iters = len(data_loader) 
        for i, (features, label) in enumerate(data_loader):
            if random.random() < 0.1:
                bsz = label.shape[0]
                item_emb = model.emb_layer['item_id']
                origin_item_emb = item_emb(features['item_id']).squeeze(1)
                mean_item_emb = torch.mean(item_emb.weight.data, dim=0, keepdims=True) \
                                    .repeat(bsz, 1)
                y = model.forward_with_item_id_emb(mean_item_emb, features)
            else:
                y = model(features)
            loss = criterion(y, label.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                print("    iters {}/{} loss: {:.4f}".format(i + 1, total_iters + 1, total_loss/log_interval), end='\r')
                total_loss = 0

        print("Epoch {}/{} loss: {:.4f}".format(epoch_i, epoch, epoch_loss/total_iters), " " * 20)
    return 

def train(model, data_loader, device, epoch, lr, weight_decay, save_path, log_interval=10, val_data_loader=None):
    # train
    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), \
                                            lr=lr, weight_decay=weight_decay)
    for epoch_i in range(1, epoch + 1):
        epoch_loss = 0.0
        total_loss = 0
        total_iters = len(data_loader) 
        for i, (features, label) in enumerate(data_loader):
            y = model(features)
            loss = criterion(y, label.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                print("    Iter {}/{} loss: {:.4f}".format(i + 1, total_iters + 1, total_loss/log_interval), end='\r')
                total_loss = 0
    return 

def pretrain(dataset_name, 
         datahub_name,
         bsz,
         shuffle,
         model_name,
         epoch,
         lr,
         weight_decay,
         device,
         save_dir,
         is_dropoutnet=False):
    device = torch.device(device)
    save_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dataloaders = get_loaders(dataset_name, datahub_name, device, bsz, shuffle==1)
    model = get_model(model_name, dataloaders).to(device)
    save_path = os.path.join(save_dir, 'model.pth')
    print("="*20, 'pretrain {}'.format(model_name), "="*20)
    # init parameters
    model.init()
    # pretrain
    if is_dropoutnet:
        dropoutNet_train(model, dataloaders['train_base'], device, epoch, lr, weight_decay, save_path, val_data_loader=dataloaders['test'])
    else:
        train(model, dataloaders['train_base'], device, epoch, lr, weight_decay, save_path, val_data_loader=dataloaders['test'])
    print("="*20, 'pretrain {}'.format(model_name), "="*20)
    return model, dataloaders

def base(model,
         dataloaders,
         model_name,
         epoch,
         lr,
         weight_decay,
         device,
         save_dir):
    print("*"*20, "base", "*"*20)
    device = torch.device(device)
    save_dir = os.path.join(save_dir, model_name)
    save_path = os.path.join(save_dir, 'model.pth')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # data set list
    auc_list = []
    f1_list = []
    dataset_list = ['train_warm_a', 'train_warm_b', 'train_warm_c', 'test']
    for i, train_s in enumerate(dataset_list):
        auc, f1 = test(model, dataloaders['test'], device)
        auc_list.append(auc)
        f1_list.append(f1)
        print("[base model] evaluate on [test dataset] auc: {:.4f}, F1 socre: {:.4f}".format(auc, f1))
        if i < 3:
            model.only_optimize_itemid()
            train(model, dataloaders[train_s], device, epoch, lr, weight_decay, save_path)
    print("*"*20, "base", "*"*20)
    return auc_list, f1_list

def metaE(model,
          dataloaders,
          model_name,
          epoch,
          lr,
          weight_decay,
          device,
          save_dir):
    print("*"*20, "metaE", "*"*20)
    device = torch.device(device)
    save_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'model.pth')
    train_base = dataloaders['train_base']
    metaE_model = MetaE(model, warm_features=dataloaders.item_features, device=device).to(device)
    # fetch data
    metaE_dataloaders = [dataloaders[name] for name in ['metaE_a', 'metaE_b', 'metaE_c', 'metaE_d']]
    # train meta embedding generator
    metaE_model.train()
    criterion = torch.nn.BCELoss()
    metaE_model.optimize_metaE()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, metaE_model.parameters()), \
                                            lr=lr, weight_decay=weight_decay)
    for epoch_i in range(epoch):
        dataloader_a = metaE_dataloaders[epoch_i]
        dataloader_b = metaE_dataloaders[(epoch_i + 1) % 4]
        epoch_loss = 0.0
        total_iter_num = len(dataloader_a)
        iter_dataloader_b = iter(dataloader_b)
        for i, (features_a, label_a) in enumerate(dataloader_a):
            features_b, label_b = next(iter_dataloader_b)
            loss_a, target_b = metaE_model(features_a, label_a, features_b, criterion)
            loss_b = criterion(target_b, label_b.float())
            loss = 0.1 * loss_a + 0.9 * loss_b
            metaE_model.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            if (i + 1) % 10 == 0:
                print("    iters {}/{}, loss: {:.4f}, loss_a: {:.4f}, loss_b: {:.4f}".format(i + 1, int(total_iter_num), loss, loss_a, loss_b), end='\r')
        print("Epoch {}/{} loss: {:.4f}".format(epoch_i, epoch, epoch_loss/total_iter_num), " " * 100)
    # replace item id embedding with warmed itemid embedding
    train_a = dataloaders['train_warm_a']
    for (features, label) in train_a:
        origin_item_id_emb = metaE_model.model.emb_layer[metaE_model.item_id_name].weight.data
        warm_item_id_emb = metaE_model.warm_item_id(features)
        indexes = features[metaE_model.item_id_name].squeeze()
        origin_item_id_emb[indexes, ] = warm_item_id_emb
    # test by steps 
    dataset_list = ['train_warm_a', 'train_warm_b', 'train_warm_c', 'test']
    auc_list = []
    f1_list = []
    for i, train_s in enumerate(dataset_list):
        print("#"*10, dataset_list[i],'#'*10)
        train_s = dataset_list[i]
        auc, f1 = test(metaE_model.model, dataloaders['test'], device)
        auc_list.append(auc.item())
        f1_list.append(f1.item())
        print("[metaE] evaluate on [test dataset] auc: {:.4f}, F1 score: {:.4f}".format(auc, f1))
        if i < len(dataset_list) - 1:
            metaE_model.model.only_optimize_itemid()
            train(metaE_model.model, dataloaders[train_s], device, epoch, lr, weight_decay, save_path)
    print("*"*20, "metaE", "*"*20)
    return auc_list, f1_list

def mwuf(model,
         dataloaders,
         model_name,
         epoch,
         lr,
         weight_decay,
         device,
         save_dir):
    print("*"*20, "mwuf", "*"*20)
    device = torch.device(device)
    save_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'model.pth')
    train_base = dataloaders['train_base']
    # train mwuf
    mwuf_model = MWUF(model, 
                      item_features=dataloaders.item_features,
                      train_loader=train_base,
                      device=device).to(device)
    
    mwuf_model.init_meta()
    mwuf_model.train()
    criterion = torch.nn.BCELoss()
    mwuf_model.optimize_new_item_emb()
    optimizer1 = torch.optim.Adam(params=filter(lambda p: p.requires_grad, mwuf_model.parameters()), \
                                            lr=lr, weight_decay=weight_decay)
    mwuf_model.optimize_meta()
    optimizer2 = torch.optim.Adam(params=filter(lambda p: p.requires_grad, mwuf_model.parameters()), \
                                            lr=lr, weight_decay=weight_decay)
    mwuf_model.optimize_all()
    total_iters = len(train_base)
    loss_1, loss_2 = 0.0, 0.0
    for i, (features, label) in enumerate(train_base):
        # if i + 1 > total_iters * 0.3:
        #     break
        y_cold = mwuf_model.cold_forward(features)
        cold_loss = criterion(y_cold, label.float())
        mwuf_model.zero_grad()
        cold_loss.backward()
        optimizer1.step()
        y_warm = mwuf_model.forward(features)
        warm_loss = criterion(y_warm, label.float())
        mwuf_model.zero_grad()
        warm_loss.backward()
        optimizer2.step()
        loss_1 += cold_loss
        loss_2 += warm_loss
        if (i + 1) % 10 == 0:
            print("    iters {}/{}  warm loss: {:.4f}" \
                    .format(i + 1, int(total_iters), \
                     warm_loss.item()), end='\r')
    print("final average warmup loss: cold-loss: {:.4f}, warm-loss: {:.4f}"
                    .format(loss_1/total_iters, loss_2/total_iters))
    # use trained meta scale and shift to initialize embedding of new items
    train_a = dataloaders['train_warm_a']
    for (features, label) in train_a:
        origin_item_id_emb = mwuf_model.model.emb_layer[mwuf_model.item_id_name].weight.data
        warm_item_id_emb = mwuf_model.warm_item_id(features)
        indexes = features[mwuf_model.item_id_name].squeeze()
        origin_item_id_emb[indexes, ] = warm_item_id_emb
    # test by steps 
    dataset_list = ['train_warm_a', 'train_warm_b', 'train_warm_c', 'test']
    auc_list = []
    f1_list = []
    for i, train_s in enumerate(dataset_list):
        print("#"*10, dataset_list[i],'#'*10)
        train_s = dataset_list[i]
        auc, f1 = test(mwuf_model.model, dataloaders['test'], device)
        auc_list.append(auc.item())
        f1_list.append(f1.item())
        print("[mwuf] evaluate on [test dataset] auc: {:.4f}, F1 score: {:.4f}".format(auc, f1))
        if i < len(dataset_list) - 1:
            mwuf_model.model.only_optimize_itemid()
            train(mwuf_model.model, dataloaders[train_s], device, epoch, lr, weight_decay, save_path)
    print("*"*20, "mwuf", "*"*20)
    return auc_list, f1_list


def creu(model,
         dataloaders,
         model_name,
         num_components,
         epoch,
         creu_epochs,
         creu_iters,
         lr,
         weight_decay,
         device,
         save_dir,
         only_init=False):
    print("*" * 20, "CREU", "*" * 20)
    device = torch.device(device)
    save_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'model.pth')
    train_base = dataloaders['train_base']

    warm_model = CREU(model,
                      warm_features=dataloaders.item_features,
                      train_loader=train_base,
                      num_components=num_components,
                      device=device).to(device)
    warm_model.init_cvae()

    def compute_metrics(model, dataloader, device):
        model.eval()
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for features, labels in tqdm(dataloader, desc="Evaluating", position=0, leave=True):
                target, uncertainty, _, _ = model(features)
                all_labels.append(labels.cpu().numpy())
                all_predictions.append(target.detach().cpu().numpy())

        all_labels = np.concatenate(all_labels, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        auc = roc_auc_score(all_labels, all_predictions)

        return auc

    def warm_up(dataloader, epochs, iters, phase, logger=False):
        warm_model.train()
        criterion = torch.nn.BCELoss()
        warm_model.optimize_cvae()
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, warm_model.parameters()),
                                     lr=lr, weight_decay=weight_decay)
        batch_num = len(dataloader)

        # Initialize lists to store metrics
        auc_epoch_list = []
        uncertainty_epoch_list = []

        # Train warm-up model
        for e in range(epochs):
            total_a, total_b, total_c, total_d, total_f = 0.0, 0.0, 0.0, 0.0, 0.0
            for i, (features, label) in enumerate(
                    tqdm(dataloader, desc="Training Progress", position=0, leave=True, dynamic_ncols=True)):
                a, b, c, d, f = 0.0, 0.0, 0.0, 0.0, 0.0
                for iter in range(iters):
                    target, uncertainty, recon_term, reg_term = warm_model(features)
                    main_loss = criterion(target, label.float())
                    # loss = main_loss + recon_term + 1e-4 * reg_term + uncertainty
                    loss = main_loss + recon_term + 1e-4 * reg_term
                    warm_model.zero_grad()
                    loss.backward()
                    optimizer.step()

                    a, b, c, d, f = a + loss.item(), b + main_loss.item(), c + recon_term.item(), d + reg_term.item(), f + uncertainty.item()
                a, b, c, d, f = a / iters, b / iters, c / iters, d / iters, f / iters
                total_a, total_b, total_c, total_d, total_f = total_a + a, total_b + b, total_c + c, total_d + d, total_f + f

            # Calculate AUC and uncertainty for each epoch
            auc = compute_metrics(warm_model, dataloader, device)
            auc_epoch_list.append(auc)
            uncertainty_epoch_list.append(total_f / batch_num)

            if logger and iters != 1:
                log_message = "loss: {:.4f}, main loss: {:.4f}, recon loss: {:.4f}, reg loss: {:.4f}, uncertainty: {:.8f}".format(
                    a, b, c, d, f)
                tqdm.write(log_message)

            print(
                "\nEpoch {}/{} loss: {:.4f}, main loss: {:.4f}, recon loss: {:.4f}, reg loss: {:.4f}, uncertainty: {:.8f}".format(
                    e, epochs, total_a / batch_num, total_b / batch_num, total_c / batch_num, total_d / batch_num,
                               total_f / batch_num), " " * 100
            )

        # Log AUC and uncertainty per epoch
        for epoch_idx in range(epochs):
            print(
                f"[Epoch {epoch_idx + 1}] AUC: {auc_epoch_list[epoch_idx]:.4f}, Uncertainty: {uncertainty_epoch_list[epoch_idx]:.8f}")

        # Save AUC and uncertainty metrics to a file
        metrics_save_path = os.path.join(save_dir, f"metrics_epoch_{phase}.txt".format(epoch))
        with open(metrics_save_path, 'w') as f:
            for epoch_idx in range(epochs):
                f.write(
                    f"Epoch {epoch_idx + 1} - AUC: {auc_epoch_list[epoch_idx]:.4f}, Uncertainty: {uncertainty_epoch_list[epoch_idx]:.8f}\n")

        return auc_epoch_list, uncertainty_epoch_list

    warm_up(train_base, epochs=10, iters=creu_iters, logger=True, phase="cold")

    # Evaluate the model at different phases
    dataset_list = ['train_warm_a', 'train_warm_b', 'train_warm_c', 'test']
    auc_list, f1_list = [], []
    for i, train_s in enumerate(dataset_list):
        print("#" * 10, dataset_list[i], '#' * 10)
        auc, f1 = test(warm_model.model, dataloaders['test'], device)
        auc_list.append(auc)
        f1_list.append(f1)
        print("[CREU] evaluate on [test dataset] auc: {:.4f}, F1 score: {:.4f}".format(auc, f1))

        if i < len(dataset_list) - 1:
            warm_model.model.only_optimize_itemid()
            train(warm_model.model, dataloaders[train_s], device, epoch, lr, weight_decay, save_path)
            if not only_init:
                warm_up(dataloaders[train_s], iters=creu_iters, epochs=creu_epochs, phase=f"warm_{i}")

    print("*" * 20, "CREU", "*" * 20)
    return auc_list, f1_list


def run(model, dataloaders, args, model_name, warm):
    if warm == 'base':
        auc_list, f1_list = base(model, dataloaders, model_name, args.epoch, args.lr, args.weight_decay, args.device, args.save_dir)
    elif warm == 'mwuf':
        auc_list, f1_list = mwuf(model, dataloaders, model_name, args.epoch, args.lr, args.weight_decay, args.device, args.save_dir)
    elif warm == 'metaE': 
        auc_list, f1_list = metaE(model, dataloaders, model_name, args.epoch, args.lr, args.weight_decay, args.device, args.save_dir)
    elif warm == 'creu':
        auc_list, f1_list = creu(model, dataloaders, model_name, args.num_components, args.epoch, args.creu_epochs, args.creu_iters, args.lr, args.weight_decay, args.device, args.save_dir)
    return auc_list, f1_list

if __name__ == '__main__':
    args = get_args()
    if args.seed > -1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    res = {}
    print(args.model_name)
    torch.cuda.empty_cache()
    # load or train pretrain models
    drop_suffix = '-dropoutnet' if args.is_dropoutnet else ''
    model_path = os.path.join(args.pretrain_model_path, args.model_name + drop_suffix + '-{}-{}'.format(args.dataset_name, args.seed))
    if os.path.exists(model_path):
        model = torch.load(model_path, weights_only=False).to(args.device)
        dataloaders = get_loaders(args.dataset_name, args.datahub_path, args.device, args.bsz, args.shuffle==1)
    else:
        model, dataloaders = pretrain(args.dataset_name, args.datahub_path, args.bsz, args.shuffle, args.model_name, \
            args.epoch, args.lr, args.weight_decay, args.device, args.save_dir, args.is_dropoutnet)
        if len(args.pretrain_model_path) > 0:
            torch.save(model, model_path)
    # warmup train and test
    avg_auc_list, avg_f1_list = [], []
    for i in range(args.runs):
        model_v = copy.deepcopy(model).to(args.device)
        auc_list, f1_list = run(model_v, dataloaders, args, args.model_name, args.warmup_model)
        avg_auc_list.append(np.array(auc_list))
        avg_f1_list.append(np.array(f1_list))
    avg_auc_list = list(np.stack(avg_auc_list).mean(axis=0))
    avg_f1_list = list(np.stack(avg_f1_list).mean(axis=0))
    print("auc: {}".format(avg_auc_list))
    print("f1: {}".format(avg_f1_list))
    try:
        max_auc_list = np.stack(avg_auc_list).max(axis=0).tolist()
        max_f1_list = np.stack(avg_f1_list).max(axis=0).tolist()
        print("max auc: {}".format(max_auc_list))
        print("max f1: {}".format(max_f1_list))
    except:
        pass

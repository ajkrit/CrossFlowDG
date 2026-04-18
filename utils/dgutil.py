import torch
import os


def save_results_fast(logits, limgs, ltxts, xss, ys, filename="results.pt"):

    all_logits = torch.cat([t.cpu() for t in logits], dim=0)
    all_imgs = torch.cat([t.cpu() for t in limgs], dim=0)
    all_txts = torch.cat([t.cpu() for t in ltxts], dim=0)
    all_ys   = torch.cat([t.cpu() for t in ys], dim=0)

    batch_trajectories = [torch.stack(batch_xs).cpu() for batch_xs in xss]
    all_trajectories = torch.cat(batch_trajectories, dim=1)

    data = {
        "logits": all_logits,      # [N, C]
        "limgs": all_imgs,         # [N, Dim]
        "ltxts": all_txts,         # [N, Dim]
        "ys":    all_ys,           # [N]
        "xss":   all_trajectories  # [Steps, N, Dim]
    }

    print(f"Saving to {filename}...")
    torch.save(data, filename)
    print("Done!")


def train_valid_target_eval_names(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    t = 0    
    '''t represent the index of the dataloader in eval_loader, e.g., eval_loader = [0, 1, 2, 0, 1, 2, 3], 
    where 0-4 is the proxy of the domian, the 4-th domain is the target domain. the first three 0 1 2 is the source domain, 
    while the latter 0 1 2 is the valid-set, and the last 3 is the target domain'''
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['train'].append(t)
            t += 1
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['valid'].append(t)
        else:
            eval_name_dict['target'].append(t)
        t += 1
    return eval_name_dict

def DG_accuracy(config, logger, model, loader, item, epoch):
    correct = 0
    total = 0

    logits = []
    limgs = []
    ltxts = []
    xss = []
    ys = []

    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(loader):
            x = data[0].cuda().float()
            input_ids = data[1]['input_ids'].long()
            attention_mask = data[1]['attention_mask'].long()
            y = data[2].cuda().long()
            p, xs, limg, ltxt = model(x, input_ids, attention_mask)

            if p.size(1) == 1:
                correct += (p.gt(0).ep(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)

            if item in ['valid', 'target']:
                logits.append(p)
                limgs.append(limg)
                ltxts.append(ltxt)
                xss.append(xs)
                ys.append(y)
        
        if item in ['valid', 'target']:
            path = f"{config.OUTPUT}/{item}_data"
            os.makedirs(path, exist_ok=True)
            save_results_fast(logits, limgs, ltxts, xss, ys, f"{path}/data_{epoch}_{idx}.pt")

    model.train()
    return correct / total


def DG_target_accuracy(config, logger, model, loader):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(loader):
            x = data[0].cuda().float()
            y = data[2].cuda().long()
            p, _, _, _ = model(x)

            if p.size(1) == 1:
                correct += (p.gt(0).ep(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
        
    model.train()
    return correct / total


def img_param_init(args):
    dataset = args.dataset
    if dataset == 'PACS':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    elif dataset == 'VLCS':
        domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    elif dataset == 'office-home':
        domains = ['Art', 'Clipart', 'Product', 'Real_World']
    elif dataset == 'terra_incognita':
        domains = ['location_100', 'location_38', 'location_43', 'location_46']
    elif dataset == 'domain_net':
        domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    else:
        print('No such dataset exists!')
    args.domains = domains
    args.img_dataset = {
        'PACS': ['art_painting', 'cartoon', 'photo', 'sketch'], 
        'VLCS': ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007'], 
        'office-home': ['Art', 'Clipart', 'Product', 'Real_World'], 
        'terra_incognita': ['location_100', 'location_38', 'location_43', 'location_46'], 
        'domain_net': ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    }
    args.input_shape = (3, 224, 224)
    if args.dataset == 'PACS':
        args.num_classes = 7
    elif dataset == 'VLCS':
        args.num_classes = 5
    elif args.dataset == 'office-home':
            args.num_classes = 65
    elif args.dataset == 'terra_incognita':
        args.num_classes = 10
    elif args.dataset == 'domain_net':
        args.num_classes = 345
    return args
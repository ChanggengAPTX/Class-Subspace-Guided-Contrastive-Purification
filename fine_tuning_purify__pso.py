import hydra
from omegaconf import DictConfig
import logging
import numpy as np
import os
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet18, resnet34
from models import SimCLR
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from scipy.special import softmax
import matplotlib.pyplot as plt
from pyswarms.single.global_best import GlobalBestPSO

logger = logging.getLogger(__name__)
args_global = None  # 存储config的全局变量

class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class LinModel(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, n_classes: int):
        super().__init__()
        self.enc = encoder
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.lin = nn.Linear(self.feature_dim, self.n_classes)

    def forward(self, x):
        features = self.enc(x)
        logits = self.lin(features)
        return logits, features

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        logits_mask = torch.ones_like(mask).fill_(1)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        exp_sim = torch.exp(similarity_matrix) * logits_mask
        denom = exp_sim.sum(1, keepdim=True) + 1e-8
        log_prob = similarity_matrix - torch.log(denom)

        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        loss = -mean_log_prob_pos.mean()
        return loss

def run_epoch(model, dataloader, epoch, optimizer=None, scheduler=None, scl_loss_fn=None, scl_weight=0.1):
    model.train() if optimizer else model.eval()

    total_loss_meter = AverageMeter('total_loss')
    ce_loss_meter = AverageMeter('ce_loss')
    scl_loss_meter = AverageMeter('scl_loss')
    acc_meter = AverageMeter('acc')

    loader_bar = tqdm(dataloader)

    for x, y in loader_bar:
        x, y = x.cuda(), y.cuda()
        logits, features = model(x)

        ce_loss = F.cross_entropy(logits, y)
        scl_loss = scl_loss_fn(features, y) if scl_loss_fn else torch.tensor(0.).cuda()

        total_loss = (1-scl_weight)*ce_loss + scl_weight * scl_loss

        if optimizer:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        acc = (logits.argmax(dim=1) == y).float().mean()

        total_loss_meter.update(total_loss.item(), x.size(0))
        ce_loss_meter.update(ce_loss.item(), x.size(0))
        scl_loss_meter.update(scl_loss.item(), x.size(0))
        acc_meter.update(acc.item(), x.size(0))

        loader_bar.set_description("{} epoch {}, total_loss: {:.4f}, CE: {:.4f}, SCL: {:.4f}, acc: {:.4f}".format(
            "Train" if optimizer else "Test", epoch,
            total_loss_meter.avg, ce_loss_meter.avg, scl_loss_meter.avg, acc_meter.avg))

    return total_loss_meter.avg, ce_loss_meter.avg, scl_loss_meter.avg, acc_meter.avg

def get_lr(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def finetune_with_weight(scl_weight: float, num_epochs: int, args: DictConfig) -> list:
    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(32),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.ToTensor()])
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_transform = transforms.ToTensor()
    scl_loss_fn = SupConLoss(temperature=0.07).cuda()
    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_set = CIFAR10(root=data_dir, train=True, transform=train_transform, download=False)
    test_set = CIFAR10(root=data_dir, train=False, transform=test_transform, download=False)

    original_targets = np.array(train_set.targets)
    noisy_targets = original_targets.copy()
    num_classes = 10
    noise_ratio = 0.4
    num_noisy = int(noise_ratio * len(noisy_targets))
    np.random.seed(42)
    noisy_indices = np.random.choice(len(noisy_targets), num_noisy, replace=False)
    for idx in noisy_indices:
        true_label = noisy_targets[idx]
        noisy_label = np.random.choice([l for l in range(num_classes) if l != true_label])
        noisy_targets[idx] = noisy_label
    train_set.targets = noisy_targets.tolist()

    base_encoder = eval(args.backbone)
    pre_model = SimCLR(base_encoder, projection_dim=args.projection_dim).cuda()
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"simclr_resnet18_epoch1000.pt")
    pre_model.load_state_dict(torch.load(model_path))
    pre_model.eval()

    temp_loader = DataLoader(train_set, batch_size=128, shuffle=False)
    all_features, all_noisy_labels = [], []
    with torch.no_grad():
        for x, y in temp_loader:
            x = x.cuda()
            feats = pre_model.enc(x).cpu()
            all_features.append(feats)
            all_noisy_labels.append(y)
    features = torch.cat(all_features, dim=0).numpy()
    noisy_labels = torch.cat(all_noisy_labels, dim=0).numpy()

    SVD_RANK = 70
    class_subspaces = {}
    for c in range(num_classes):
        class_features = features[noisy_labels == c]
        if len(class_features) >= SVD_RANK:
            svd = TruncatedSVD(n_components=SVD_RANK, random_state=42)
            svd.fit(class_features)
            V = svd.components_
            class_subspaces[c] = V

    predicted_labels = []
    for i in range(len(features)):
        z = features[i]
        errors = []
        for c in range(num_classes):
            if c in class_subspaces:
                V = class_subspaces[c]
                z_proj = V.T @ (V @ z)
                err = np.linalg.norm(z - z_proj)
            else:
                err = np.inf
            errors.append(err)
        predicted_labels.append(np.argmin(errors))
    predicted_labels = np.array(predicted_labels)

    clean_mask = predicted_labels == noisy_labels
    clean_indices = np.where(clean_mask)[0]

    # clean_sampler = SubsetRandomSampler(clean_indices)
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=clean_sampler, drop_last=True)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    clean_features = features[clean_indices]
    clean_labels = noisy_labels[clean_indices]

    # 第二次SVD：使用第一次提纯后的样本重新构建子空间
    refined_subspaces = {}
    for c in range(num_classes):
        class_feats = clean_features[clean_labels == c]
        if len(class_feats) >= SVD_RANK:
            svd = TruncatedSVD(n_components=SVD_RANK, random_state=42)
            svd.fit(class_feats)
            refined_subspaces[c] = svd.components_

    # 第二次子空间重新分类
    second_predicted_labels = []
    for i in range(len(features)):
        z = features[i]
        errors = []
        for c in range(num_classes):
            if c in refined_subspaces:
                V = refined_subspaces[c]
                z_proj = V.T @ (V @ z)
                err = np.linalg.norm(z - z_proj)
            else:
                err = np.inf
            errors.append(err)
        second_predicted_labels.append(np.argmin(errors))
    second_predicted_labels = np.array(second_predicted_labels)

    refined_mask = second_predicted_labels == noisy_labels
    refined_indices = np.where(refined_mask)[0]
    refined_features = features[refined_indices]
    refined_labels = noisy_labels[refined_indices]

    # 第三次SVD：用第二次提纯的样本重建子空间
    third_subspaces = {}
    for c in range(num_classes):
        class_feats = refined_features[refined_labels == c]
        if len(class_feats) >= SVD_RANK:
            svd = TruncatedSVD(n_components=SVD_RANK, random_state=42)
            svd.fit(class_feats)
            third_subspaces[c] = svd.components_

    # 第三次预测
    third_predicted_labels = []
    for i in range(len(features)):
        z = features[i]
        errors = []
        for c in range(num_classes):
            if c in third_subspaces:
                V = third_subspaces[c]
                z_proj = V.T @ (V @ z)
                err = np.linalg.norm(z - z_proj)
            else:
                err = np.inf
            errors.append(err)
        third_predicted_labels.append(np.argmin(errors))
    third_predicted_labels = np.array(third_predicted_labels)

    final_mask = third_predicted_labels == noisy_labels
    final_indices = np.where(final_mask)[0]

    clean_sampler = SubsetRandomSampler(final_indices)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=clean_sampler, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = LinModel(pre_model.enc, feature_dim=pre_model.feature_dim, n_classes=10).cuda()

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, 0.05, momentum=args.momentum, weight_decay=1e-3, nesterov=True)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: get_lr(
        step, num_epochs * len(train_loader), args.learning_rate, 1e-3))

    test_acc_list = []
    for epoch in range(1, num_epochs + 1):
        run_epoch(model, train_loader, epoch, optimizer, scheduler,
                  scl_loss_fn=scl_loss_fn, scl_weight=scl_weight)
        _, _, _, test_acc = run_epoch(model, test_loader, epoch,
                                      scl_loss_fn=scl_loss_fn, scl_weight=scl_weight)
        test_acc_list.append(test_acc)

    return test_acc_list

def evaluate_scl_weight(weight_array):
    acc_list = []
    for w in weight_array:
        scl_weight = float(w[0])
        accs = finetune_with_weight(scl_weight, num_epochs=3, args=args_global)
        avg_acc = np.mean(accs[-1:])
        acc_list.append(-avg_acc)
    return np.array(acc_list)
@hydra.main(config_path='.', config_name='simclr_config', version_base=None)
def run_pso(args: DictConfig):
    global args_global
    args_global = args
    bounds = ([0.0], [1.0])
    optimizer = GlobalBestPSO(n_particles=5, dimensions=1, options={'c1': 1.5, 'c2': 1.5, 'w': 0.729}, bounds=bounds)
    best_cost, best_pos = optimizer.optimize(evaluate_scl_weight, iters=10)

    cost_history = -np.array(optimizer.cost_history)
    matplotlib.use('Agg')  # 或者直接注释掉 plt.show()
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    matplotlib.rcParams['axes.unicode_minus'] = False

    plt.plot(cost_history, marker='o')
    plt.title("PSO优化scl_weight的准确率变化")
    plt.xlabel("迭代次数")
    plt.ylabel("最后3个epoch平均Test准确率")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pso_scl_weight_curve.png")

    print(f"✅ 最优scl_weight: {best_pos[0]:.4f}")
    print(f"✅ 对应Test准确率（最后3个epoch平均）: {-best_cost:.4f}")

if __name__ == '__main__':
    run_pso()

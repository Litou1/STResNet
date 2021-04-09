import math
from collections import Counter
from collections import defaultdict
import torch
from torch.optim.lr_scheduler import _LRScheduler


class MultiStepLR_Restart(_LRScheduler):
    def __init__(self, optimizer, milestones, restarts=None, weights=None, gamma=0.1,
                 clear_state=False, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.clear_state = clear_state
        self.restarts = restarts if restarts else [0]
        self.restart_weights = weights if weights else [1]
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(MultiStepLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            if self.clear_state:
                self.optimizer.state = defaultdict(dict)
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [
            group['lr'] * self.gamma**self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]


class CosineAnnealingLR_Restart(_LRScheduler):
    def __init__(self, optimizer, T_period, restarts=None, weights=None, eta_min=0, last_epoch=-1):
        self.T_period = [int(t) for t in T_period]
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = restarts if restarts else [0]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[self.restarts.index(self.last_epoch) + 1]
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) /
                (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]


if __name__ == "__main__":
    optimizer = torch.optim.Adam([torch.zeros(3, 64, 3, 3)], lr=2e-4, weight_decay=0,
                                 betas=(0.9, 0.99))
    ##############################
    # MultiStepLR_Restart
    ##############################
    ## Original
    lr_steps = [20e3, 40e3, 60e3, 80e3]
    restarts = None
    restart_weights = None

    # ## two

    # lr_steps = [20e3, 40e3, 80e3, 120e3, 160e3, 180e3, 200e3, 240e3, 280e3]
    # restarts = [160e3]
    # restart_weights = [1]
    #
    # ## four
    # lr_steps = [10e3, 20e3, 40e3, 60e3, 80e3, 90e3, 100e3, 120e3, 140e3, \
    #             170e3, 180e3, 200e3, 220e3, 240e3, 250e3, 260e3, 280e3, 300e3]
    # restarts = [80e3, 160e3, 240e3]
    # restart_weights = [1, 1, 1]

    scheduler = MultiStepLR_Restart(optimizer, lr_steps, restarts, restart_weights, gamma=0.5,
                                    clear_state=False)

    ##############################
    # Cosine Annealing Restart
    ##############################

    ## one
    # T_period = [160e3]
    #
    # ## four
    # T_period = [20e3, 20e3, 40e3, 80e3]
    # restarts = [20e3, 40e3, 80e3]
    # restart_weights = [1, 1, 1]

    # three
    # T_period = [1e3, 1e3, 1e3, 1e3, 2e3, 2e3, 2e3]
    # restarts = [1e3, 2e3, 3e3, 4e3, 6e3, 8e3]
    # restart_weights = [1, 1, 1, 1, 1, 1]
    # eta_min = 2e-6
    # scheduler = CosineAnnealingLR_Restart(optimizer, T_period, eta_min=eta_min, restarts=restarts,
    #                                       weights=restart_weights)

    ##############################
    # Draw figure
    ##############################
    N_iter = int(100e3)
    lr_l = list(range(N_iter))
    for i in range(N_iter):
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        lr_l[i] = current_lr

    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import matplotlib.ticker as mtick
    mpl.style.use('default')
    import seaborn
    seaborn.set(style='whitegrid')
    seaborn.set_context('paper')

    plt.figure(1)
    plt.subplot(111)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.title('Title', fontsize=16, color='k')
    plt.plot(list(range(N_iter)), lr_l, linewidth=1.5, label='learning rate scheme')
    legend = plt.legend(loc='upper right', shadow=False)
    ax = plt.gca()
    labels = ax.get_xticks().tolist()
    for k, v in enumerate(labels):
        labels[k] = str(int(v / 1000)) + 'K'
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    # plt.yscale('log')

    ax.set_ylabel('Learning rate')
    ax.set_xlabel('Iteration')
    fig = plt.gcf()
    plt.show()
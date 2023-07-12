import torch
import torch.optim as optim

def get_sgd_optimizer(model):
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    return optimizer

def get_one_cycle_LR_scheduler(optimizer, train_loader, best_lr, epochs):
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=best_lr, 
                                                    steps_per_epoch=len(train_loader),
                                                    pct_start=5/epochs, div_factor = 25, 
                                                    three_phase=False, epochs=epochs, 
                                                    anneal_strategy='linear', final_div_factor=25,
                                                    verbose=False)
    return scheduler

def get_adam_optimizer(model):
    #optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
    optimizer = optim.Adam(model.parameters(), lr=1e-9, weight_decay=1e-2)
    return optimizer
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import os
from tqdm import tqdm
import random
from pathlib import Path

from nn_model import MyNN
from nn_data_preparation import DataPreparation
from config import config


random.seed(config.general.seed)
np.random.seed(config.general.seed)
torch.manual_seed(config.general.seed)
torch.cuda.manual_seed_all(config.general.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():

    train_loader, val_loader, test_loader = DataPreparation(train=True).prepare_data(pd.read_csv(Path(config.paths.path_to_train_data)))

    #объявляем модель
    model = MyNN(train_loader.dataset[0][0].shape[0], 1).to(device)
    
    # задаем оптимизатор
    opt_class = getattr(torch.optim, config.optimizer.name)
    opt = opt_class(model.parameters(), lr=config.optimizer.params.lr, weight_decay=config.optimizer.params.weight_decay)

    # задаем loss
    loss_class = getattr(nn, config.loss.name)
    loss_reg = loss_class()

    # задаем шедулер
    scheduler_class = getattr(torch.optim.lr_scheduler, config.scheduler.name)
    lr_scheduler = scheduler_class( opt,  #оптимизатор
                                    mode = config.scheduler.params.mode,  #'max' или 'min' - следим чтобы отслеживаемый параметр увеличивался()
                                    factor = config.scheduler.params.factor,  #коэфициэнт, на который будет умножен lr
                                    patience = config.scheduler.params.patience, #кол-во эпох без улучшения отслеживаемого пораметров
                                    eps = config.scheduler.params.eps,
                                    )

    #начинаем процесс обучения
    train_loss = []
    val_loss = []
    lr_list = []
    best_loss = None
    best_model = 0
    count = 0

    for epoch in range(config.training.num_epochs):
        model.train()
        train_loop = tqdm(train_loader,leave=False)
        running_train_loss = []
        for x,targets in train_loop:

            pred = model(x)
            loss = torch.sqrt(loss_reg(pred, targets))

            opt.zero_grad()
            loss.backward()

            opt.step()

            running_train_loss.append(loss.item())
            mean_train_loss = sum(running_train_loss)/len(running_train_loss)

            train_loop.set_description(f'Epoch[{epoch+1}/{config.training.num_epochs}], train_loss = {mean_train_loss:.4f}')

        train_loss.append(mean_train_loss)

        model.eval()
        with torch.no_grad():
            running_val_loss = []
            for x, targets in val_loader:
                pred = model(x)
                loss = torch.sqrt(loss_reg(pred,targets))

                running_val_loss.append(loss.item())
                mean_val_loss = sum(running_val_loss)/len(running_val_loss)

            val_loss.append(mean_val_loss)

        lr_scheduler.step(mean_val_loss)

        lr = lr_scheduler._last_lr[0]
        lr_list.append(lr)
        print((f'Epoch [{epoch+1}/{config.training.num_epochs}], train_loss = {mean_train_loss:.4f},  val_loss = {mean_val_loss:.4f}, lr = {lr:.4f}'))

        if best_loss is None:
            best_loss = mean_val_loss

        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            count = 0

            if best_model == 0:
                torch.save(model.state_dict(), f'{Path(config.paths.path_to_best_nn_model)}/best_model.pt')
                best_model = epoch+1

            else:
                os.remove(f'{Path(config.paths.path_to_best_nn_model)}/best_model.pt')
                torch.save(model.state_dict(), f'{Path(config.paths.path_to_best_nn_model)}/best_model.pt')
                best_model = epoch+1

            print(f'На эпохе - {epoch+1}, сохранена модель со значением функции потерь на валидации - {best_loss:.4f}', end='\n\n') 

        if count >= config.training.early_stopping_epochs:
            print(f'\033[31Обучение остановлено на {epoch+1} эпохе.\033[0m')
            break
        count += 1

    # загружаем веса лучшей модели и начинаем процесс тестирования
    model.load_state_dict(torch.load(f'{config.paths.path_to_best_nn_model}/best_model.pt'))
    model.eval()
    
    with torch.no_grad():
        running_test_loss = []
        for x, targets in test_loader:
            pred = model(x)
            loss = torch.sqrt(loss_reg(pred,targets))
            running_test_loss.append(loss.item())

        mean_test_loss = sum(running_test_loss)/len(running_test_loss)
        print(f'Финальный loss = {mean_test_loss:.4f}')
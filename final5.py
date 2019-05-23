import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import glob, os
from tqdm import tqdm
from torchvision import models
import torchvision.transforms as transforms
# import objgraph
import gc

is_cuda = torch.cuda.is_available()

def Fit(val_set=None, learning_rate=.00005, n_epochs=10, batch_size=6, patience = 5, learning_rate_decay = None):
    """
    print("inicio de fit...")
    print("Debut de fit...")
    print("Αρχή του fit...")
    """
    model = UNet(val_set[0].shape[-1])
    optimizer = optim.Adam([p for p in model.parameters()], lr=learning_rate)
    criterion = loss()

    if is_cuda:
        model.cuda()
        criterion.cuda()

    val_data, val_labels = val_set
    val_masks = val_data[:, 3, :, :][:, None, :, :]
    val_masks[val_masks != 0] = 1
    val_masks = torch.cat((val_masks, val_masks, val_masks), dim=1)

    X_val = val_data[:, :3, :, :]
    y_val = val_labels[:, :3, :, :]
    if is_cuda:
        X_val, y_val, M_val = Variable(X_val.cuda()), Variable(y_val.cuda()), Variable(val_masks.cuda())
    else:
        X_val, y_val, M_val = Variable(X_val), Variable(y_val), Variable(val_masks)
    N_val = X_val.shape[0]
    epoch = 0
    train_loss = []
    validation_loss = []
    file2 = 'total_dest_1000_images5.pt'
    file = '1000_images5.pt'
    labels = torch.load(file).to('cuda')
    data = torch.load(file2).to('cuda')
    train_data, train_labels = data, labels
    old_val_error = 99999999
    patience_counter = 0
    val_error_history = []
    while epoch < n_epochs:
        running_loss = 0.0
        print("iterating over the files in the folder")
        """
        for file in tqdm(os.listdir()):
            if 'total' in file:
                continue
            if '1000' not in file:
                continue
            print("here we are working with: {}".format(file))
            print("loading train data...")
            file2 = 'total_dest_' + file
            labels = torch.load(file).to('cuda')
            data = torch.load(file2).to('cuda')
            train_data, train_labels = data, labels

        """
        

        """SHUFFLING DATA"""
        print("shuffling data...")

        r = torch.randperm(train_data.shape[0]).cuda()
        train_data = train_data[r]
        masks = train_data[:, 3, :, :][:, None, :, :]
        X_train = train_data[:, :3, :, :]


        train_labels = train_labels[r]
        y_train = train_labels[:, :3, :, :]

        print("EXTRACTING THE MASKS")
        masks[masks != 0] = 1


        masks = torch.cat((masks, masks, masks), dim=1)


        if is_cuda:
            X_train, y_train, M_train = Variable(X_train.cuda()), Variable(y_train.cuda()), Variable(masks.cuda())
        else:
            X_train, y_train, M_train = Variable(X_train), Variable(y_train), Variable(masks)

        N = X_train.shape[0]

        """LOOPING OVER THE BATCHES"""
        print("now we loop...")
        model.train()
        for j in tqdm(range(int(N // batch_size))):
            j_start = j * batch_size
            j_end = (j + 1) * batch_size
            inds = range(j_start, j_end)
            X = X_train[inds]
            y = y_train[inds]
            M = M_train[inds]

            del inds

            optimizer.zero_grad()
            outputs = model(X, M)
            loss_value = criterion(Igt=y, Iout=outputs[0], mask=M)

            loss_value.backward()
            optimizer.step()

            running_loss += loss_value.item()

            del X
            del y
            del M
            del outputs
            torch.cuda.empty_cache()

            gc.collect()

        epoch_train_loss = float(running_loss) / (1000.0)
        train_loss.append(epoch_train_loss)
        del X_train
        del y_train
        del M_train

        torch.cuda.empty_cache()
        print("=" * 30)
        print("train_loss", epoch_train_loss)
        print("=" * 30)
        model.eval()

        torch.cuda.empty_cache()
        print("EVALUATING THE VALIDATION SET")
        summed_val_error = 0
        for j in tqdm(range(int(N_val // batch_size))):
            j_start = j * batch_size
            j_end = (j + 1) * batch_size
            inds = range(j_start, j_end)
            y_val_batch = y_val[inds]
            X_val_batch = X_val[inds]
            M_val_batch = M_val[inds]
            val_outputs = model(X_val_batch, M_val_batch)
            val_loss = criterion(Igt=y_val_batch, Iout=val_outputs[0], mask=M_val_batch)

            summed_val_error += val_loss.item()
            del X_val_batch
            del y_val_batch
            del M_val_batch
            del val_outputs
            torch.cuda.empty_cache()
        final_val_loss = float(summed_val_error) / (100.0)
        validation_loss.append((float(summed_val_error) / (100.0)))

        """PATIENCE, IF SET TO NONE THEN ITS OFF"""

        if patience is not None:
            if summed_val_error <= old_val_error:
                old_val_error = summed_val_error.copy()
                val_error_history.append(old_val_error)
                print("SAVING THE MODEL FOR EPOCH: {}".format(epoch + 1))

                torch.save(model.state_dict(), 'noreg_checkpoint_last_loss_epoch{}.pt'.format(epoch+1))
                torch.save(validation_loss, 'noreg_val_last_loss_epoch{}.pt'.format(epoch+1))
                torch.save(train_loss, 'noreg_train_last_loss_epoch{}.pt'.format(epoch+1))
                if len(val_error_history) == 6:
                    del val_error_history[0]
            elif summed_val_error > old_val_error:
                for i in val_error_history:
                    if i < summed_val_error:
                        patience_counter = 0
                        break
                    else:
                        patience_counter += 1


            # print("SAVING THE MODEL FOR EPOCH: {}".format(epoch + 1))
            #
            # torch.save(model.state_dict(), 'noreg_checkpoint_last_loss{}.pt'.format(epoch))
            # torch.save(validation_loss, 'noreg_val_last_loss{}.pt'.format(epoch))
            # torch.save(train_loss, 'noreg_train_last_loss{}.pt'.format(epoch))
        print("="*30)
        print("validation_loss", float(final_val_loss))
        print("="*30)
        print('END OF EPOCH {}'.format(epoch + 1))
        print("=" * 30)
        epoch += 1
        if learning_rate_decay is not None:
            if epoch == learning_rate_decay:
                if epoch % 10 == 0:
                    print("REDUCING THE LEARNING RATE FROM: {} TO: {}".format(learning_rate, learning_rate/10))
                    for para_group in optimizer.param_groups:
                        para_group['lr'] = learning_rate / 10

if __name__ == '__main__':
    labels = torch.load("../data/data14.pt").to('cuda')
    data = torch.load("../data/total_dest_data14.pt").to('cuda')
    val_data = data, labels
    del (data)
    del (labels)

    Fit(val_set=val_data, learning_rate=0.0002, n_epochs=100, batch_size=6, patience=5, learning_rate_decay=20)

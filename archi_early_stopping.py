# squared images
from PIL import Image
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
from lovelyLoss import loss as our_loss
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os
from tqdm import tqdm


class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(PartialConv2d, self).__init__()

        # this is the main convolution, which is going to be in charge of
        # getting the 2D convolution of the element-wise multiplication
        # between mask and input
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        torch.nn.init.xavier_uniform_(self.input_conv.weight)

        # it is needed to compute the convolution of the mask separately
        # in order to update its size in each level of the u-net

        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

        # conv is updated
        for param in self.input_conv.parameters():
            param.requires_grad = True

    def forward(self, args):

        image, mask = args

        output = self.input_conv(image * mask)

        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        zeros_mask = output_mask == 0
        sum_M = output_mask.masked_fill_(zeros_mask, 1.0)

        output_pre = (output - output_bias) / sum_M + output_bias
        output = output_pre.masked_fill_(zeros_mask, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(zeros_mask, 0.0)

        return output, new_mask


class UNet(nn.Module):
    def __init__(self, img_shape):
        super(UNet, self).__init__()

        # Todo : nb_jump = get_params()
        self.get_params(img_shape)

        # ENCODING :
        self.encoding()

        # DECODING :
        self.decoding()

    def get_params(self, img_shape):
        """
        This function is in charge of setting all the params
        :return:
        """
        if img_shape == 512:
            self.nb_jump = 3
        elif img_shape == 256:
            self.nb_jump = 3
        elif img_shape == 128:
            self.nb_jump = 3
        elif img_shape == 64:
            self.nb_jump = 3
        elif img_shape == 32:
            self.nb_jump = 3
        else:
            print("error in image size")

        i = int(np.log2(512 / img_shape))

        self.kernel_sizes = [7, 5, 5, 3, 3, 3, 3, 3]

        self.nb_layers = 8 - i
        self.nb_channels = []

        size = img_shape
        for j in range(self.nb_jump):
            size = int(size / 2)
            self.nb_channels.insert(0, size)

        for j in range(self.nb_layers - self.nb_jump):
            self.nb_channels.append(img_shape)

    def encoding(self):
        """
        Function in charge of the encoding part of the U-Net
        :return:
        """
        self.encoding_list = nn.ModuleList()
        self.encoding_list.append(
            nn.Sequential(
                PartialConv2d(3, self.nb_channels[0], kernel_size=self.kernel_sizes[0], stride=2,
                              padding=int((self.kernel_sizes[0] - 1) / 2)), nn.ReLU()))

        for j in range(1, self.nb_layers):
            self.encoding_list.append(
                nn.Sequential(
                    PartialConv2d(self.nb_channels[j - 1], self.nb_channels[j],
                                  kernel_size=self.kernel_sizes[j], stride=2,
                                  padding=int((self.kernel_sizes[j] - 1) / 2)),
                    nn.BatchNorm2d(self.nb_channels[j]),
                    nn.ReLU()))

    def decoding(self):
        """
        Function in charge of executing the decoding part of the U-Net
        :return:
        """
        self.decoding_list = nn.ModuleList()
        for j in range(self.nb_layers - 1):
            self.decoding_list.append(
                nn.Sequential(
                    PartialConv2d(self.nb_channels[self.nb_layers - j - 1] + self.nb_channels[self.nb_layers - j - 2],
                                  self.nb_channels[self.nb_layers - j - 2], kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(self.nb_channels[self.nb_layers - j - 2]),
                    nn.LeakyReLU(0.2)
                )
            )

        self.decoding_list.append(
            nn.Sequential(
                PartialConv2d(self.nb_channels[0] + 3, 3, kernel_size=3, stride=1, padding=1))
        )
        # no batch norm or relu here -> output is the reconstructed image
        # print(self.encoding_list)
        # print(self.decoding_list)

    def forward(self, x, mask):
        output_feature = []
        out = x, mask
        output_feature.append(out)
        for j in range(self.nb_layers):
            out = self.encoding_list[j][0](out)
            image = self.encoding_list[j][1](out[0])
            out = image, out[1]
            output_feature.append(out)

        # torch.cat((first_tensor, second_tensor), dimension)

        for j in range(self.nb_layers):
            nearestUpSample_image = nn.UpsamplingNearest2d(scale_factor=2)(out[0])
            nearestUpSample_mask = nn.UpsamplingNearest2d(scale_factor=2)(out[1])

            concat_image = torch.cat((output_feature[self.nb_layers - j - 1][0], nearestUpSample_image), dim=1)
            concat_mask = torch.cat((output_feature[self.nb_layers - j - 1][1], nearestUpSample_mask), dim=1)

            out = concat_image, concat_mask
            out = self.decoding_list[j][0](out)

            if j < self.nb_layers - 1:
                image = self.decoding_list[j][1](out[0])
                out = image, out[1]

        return out


# def LossAndOptimizer(learning_rate, model, real_image, output, mask):
#     loss = our_loss(img=real_image, output=output, mask=mask)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     return loss, optimizer


def Fit(model, train_set=None, val_set=None, learning_rate=.01, n_epochs=10, batch_size=10, patience=10, PATH = None):
    
    #early stopping :
    min_val_loss = np.Inf
    counter = 0
    early_stop = False
    if torch.cuda.is_available():
        model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = our_loss()

    epoch = 0
    train_loss = []
    validation_loss = []
    dir_path = "/home/anala/Programming_Part/data"

    # losses we want to save
    save_val_loss = []
    save_train_loss = []
    val_counter = 0
    while epoch < n_epochs:
        for file in tqdm(os.listdir(dir_path)):
            counter += 1
            if 'total' in file:
                continue
            file2 = 'total_dest_' + file
            labels = torch.load(file2)
            data = torch.load(file)
            train_data, train_labels = data, labels
            if val_counter == 0:
                validation = data[:100], labels[:100]
                val_data, val_labels = validation[0], validation[1]
                train_data, train_labels = data[100:], labels[100:]
                val_masks = val_data[:, 3, :, :][:, None, :, :]
                val_masks[val_masks != 0] = 1
                val_masks = torch.cat((val_masks, val_masks, val_masks), dim=1)

                X_val = val_data[:, :3, :, :]
                y_val = val_labels[:, :3, :, :]
                if torch.cuda.is_available():
                    X_val, y_val, M_val = Variable(X_val.cuda()), Variable(y_val.cuda()), Variable(val_masks.cuda())
                else:
                    X_val, y_val, M_val = Variable(X_val), Variable(y_val), Variable(val_masks)
            N = train_data.shape[0]
            running_loss = 0.0
            """SHUFFLING DATA"""
            train_data, train_labels = np.array(train_data), np.array(train_labels)
            train_data, train_labels = shuffle(train_data, train_labels)
            train_data, train_labels = torch.from_numpy(train_data), torch.from_numpy(train_labels)
            masks = train_data[:, 3, :, :][:, None, :, :]
            masks[masks != 0] = 1
            masks = torch.cat((masks, masks, masks), dim=1)
            """LOOPING OVER THE BATCHES"""
            for j in range(int(N // batch_size)):
                j_start = j * batch_size
                j_end = (j + 1) * batch_size
                inds = range(j_start, j_end)
                X = train_data[inds][:, :3, :, :]
                y = train_labels[inds][:, :3, :, :]
                M = masks[inds]
                if torch.cuda.is_available():
                    X, y, M = Variable(X.cuda()), Variable(y.cuda()), Variable(M.cuda())
                else:
                    X, y, M = Variable(X), Variable(y), Variable(M)
                optimizer.zero_grad()
                outputs = model(X, M)
                loss = criterion(Igt=y, Iout=outputs[0], mask=M)
                train_loss.append(loss)
                loss.backward()
                optimizer.step()
                running_loss += loss.data
                print(loss.data)
            
        train_loss.append(float(running_loss) / (N / batch_size))
        print("train_loss", float(running_loss) / (N / batch_size))

        val_outputs = model(X_val, M_val)
        val_loss = criterion(Igt=y_val, Iout=val_outputs[0], mask=M_val)
        validation_loss.append(val_loss)
        print("validation_loss", float(running_loss))

        # save variables and losses
        if epoch%20 == 0:
            torch.save(val_outputs, "val_out_{}.pt".format(epoch))
        save_val_loss += [val_loss]
        save_train_loss += [train_loss[-1]]
        
        #early_stopping :
        if val_loss > min_val_loss:
            counter += 1
            if counter >= patience:
                early_stop = True
                break
        else:
            min_val_loss = val_loss
            torch.save(model.state_dict(), PATH)
            counter = 0 

        #-------------------------
        
        print('epoch', epoch + 1)
        epoch += 1

    #save loss values
    torch.save(torch.FloatTensor(save_val_loss), "validation_losses.pt")
    torch.save(torch.FloatTensor(save_train_loss), "training_losses.pt")


    plt.plot(train_loss, label='train', color='b')
    plt.plot(validation_loss, label='validation')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    
    return early_stop

if __name__ == '__main__':
    # img512 = torch.rand(10, 4, 512, 512)
    # img256 = torch.rand(10, 4, 256, 256)
    # img128 = torch.rand(10, 4, 128, 128)
    # img64 = torch.rand(10, 4, 64, 64)
    # img32 = torch.rand(10, 4, 32, 32)
    #
    # list_img = []
    # list_img.append(img512)
    # list_img.append(img256)
    # list_img.append(img128)
    # list_img.append(img64)
    # list_img.append(img32)
    
    if torch.cuda.is_available():
       PATH = 'Programming_Part/checkpoint.pt' #put a relevant path here
    else : 
        PATH = 'Programming_Part/checkpoint.pt' #put a relevant path here

    # labels = torch.load('Programming_Part/data.pt')

    data = torch.load('Programming_Part/total_dest_data.pt')

    # val_data = data[:2], labels[:2]
    # train_data = data[2:], labels[2:]

    model = UNet(data[0].shape[-1])
    early_stop = Fit(model=model, learning_rate=0.001, n_epochs=10, batch_size=2, patience=10, PATH = 'Programming_Part/checkpoint.pt')
    
    if early_stop == True :
        model = UNet(data[0].shape[-1])
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model.load_state_dict(torch.load(PATH))
            model.to(device)
        else : 
            model.load_state_dict(torch.load(PATH))
    model.eval()
    
    #test     

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


class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, phase='encoding'):
        super(PartialConv2d, self).__init__()

        # this is the main convolution, which is going to be in charge of
        # getting the 2D convolution of the element-wise multiplication
        # between mask and input
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        torch.nn.init.xavier_normal_(self.input_conv.weight)


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

        del image, mask, output_pre, sum_M, zeros_mask, output_bias

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

        i = int(np.log2(256 / img_shape))

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
                PartialConv2d(self.nb_channels[0] + 3, 3, kernel_size=3, stride=1, padding=1, bias=True))
        )

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
        del output_feature, image, nearestUpSample_image, nearestUpSample_mask, concat_image, concat_mask
        return out


class loss(nn.modules.loss._WeightedLoss):

    def __init__(self):
        super(loss, self).__init__()
        self.C = 3
        self.H = 256
        self.W = 256
        self.N = self.C * self.H * self.W
        self.outputs = []
        self.Igt = None
        self.Iout = None
        self.mask = None
        self.Icomp = None

    def forward(self, Igt, Iout, mask):
        self.Igt = Igt
        self.Iout = Iout
        self.mask = mask
        self.make_comp()
        return self.loss_function()

    def make_comp(self):
        mask = self.mask.clone().type(torch.ByteTensor).cuda()
        self.Icomp = Variable(torch.where(mask, self.Igt, self.Iout), requires_grad = True)

        """mask = np.ones(X.shape, dtype=bool)
        mask[:, validation_ind] = 0

        X_train = X[mask].reshape(d*n_len, -1)"""

    def loss_function(self):
       
        loss_total = 6 * self.l_hole() + self.l_valid() + 0.05 * self.l_perc()\
                    + 120 * (self.l_style_comp() + self.l_style_out()) + 0.1 * self.l_tv()
        
        #print("hole : ", self.l_hole())
        #print("valid : ", self.l_valid())
        #print("perc : ", self.l_perc())

        #print("style comp : ", self.l_style_comp())
        #print("style out : ", self.l_style_out())

        #print("tv : ", self.l_tv())

        self.Igt = None
        self.Iout = None
        self.Icomp = None
        self.mask = None
        self.pool_out = None
        self.pool_gt = None
        self.pool_comp = None
        
        torch.cuda.empty_cache()

        return loss_total

    def l_hole(self):
        """
        Computation of one of the pixel losses: l_hole
        :return: value of l_hole
        """

        B, C, W, H = self.Igt.shape
        Nigt = self.N*B
        aux1 = (1 - self.mask)
        aux2 = self.Iout - self.Igt
        l_hole = torch.norm((aux1*aux2), p=1)/Nigt
        del aux1
        del aux2
        torch.cuda.empty_cache()
        return l_hole

    def l_valid(self):
        """
        Computation of one of the pixel losses: l_valid
        :return: value of l_valid
        """
        B, C, W, H = self.Igt.shape

        Nigt = self.N * B
        aux1 = self.Iout - self.Igt
        l_valid = torch.norm((aux1*self.mask), p=1)/Nigt
        del aux1
        torch.cuda.empty_cache()
        return l_valid

    def l_perc(self):
        """
        Computation of perceptual loss
        :return: value of l_perc
        """
        # min_img_size = 224  # The min size, as noted in the PyTorch pretrained models doc, is 224 px.

        """
        transform_pipeline = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
        """

        vgg16 = models.vgg16(pretrained=True).cuda()
        for param in vgg16.features.parameters():
            param.requires_grad = False
        vgg16.features._modules["4"].register_forward_hook(self.hook)
        vgg16.features._modules["9"].register_forward_hook(self.hook)
        vgg16.features._modules["16"].register_forward_hook(self.hook)

        self.outputs = []

        img = Variable(self.Iout, requires_grad=False)
        out = vgg16.forward(img)
        self.pool_out = list(self.outputs)

        self.outputs = []

        img = Variable(self.Igt, requires_grad=False)
        out = vgg16.forward(img)
        self.pool_gt = list(self.outputs)

        # comp
        self.outputs = []

        img = Variable(self.Icomp, requires_grad=False)
        out = vgg16.forward(img)
        self.pool_comp = list(self.outputs)

        l_perc = 0.0
        for i in range(len(self.pool_out)):
            Nigt = torch.tensor(self.pool_gt[i].shape).prod()
            diff1 = self.pool_out[i] - self.pool_gt[i]
            l1_loss1 = torch.norm(diff1, p=1)
            diff2 = self.pool_comp[i] - self.pool_gt[i]
            l1_loss2 = torch.norm(diff2, p=1)
            l_perc += (l1_loss1 + l1_loss2)/Nigt

        del img
        del out
        del Nigt
        del diff1
        del diff2
        del l1_loss2
        del l1_loss1
        self.outputs = []
        torch.cuda.empty_cache()

        return l_perc

    def hook(self, module, input, output):
        self.outputs.append(output)

    def l_style_out(self):
        """
        Compute style loss term: style out loss
        :return: value style out loss
        """
        l_style = 0.0
        for i in range(len(self.pool_gt)):
            B, C, H, W = self.pool_gt[i].shape
            Kp = 1.0 / (C * H * W* B)
            aux1 = torch.matmul(torch.transpose(self.pool_out[i],2,3),self.pool_out[i])
            aux2 = torch.matmul(torch.transpose(self.pool_gt[i],2,3),self.pool_gt[i])
            l_1 = torch.norm((aux1-aux2)*Kp, p=1)/(C**2)
            l_style += l_1

        del aux1
        del aux2
        del l_1
        torch.cuda.empty_cache()

        return l_style

    def l_style_comp(self):
        """
        Compute style loss term: style comp loss
        :return: value style comp loss
        """
        l_style = 0.0
        for i in range(len(self.pool_gt)):
            B, C, H, W = self.pool_gt[i].shape
            Kp = 1.0 / (C * H * W*B)
            aux1 = torch.matmul(torch.transpose(self.pool_comp[i], 2, 3), self.pool_comp[i])
            aux2 = torch.matmul(torch.transpose(self.pool_gt[i], 2, 3), self.pool_gt[i])
            l_1 = torch.norm((aux1 - aux2) * Kp, p=1) / (C ** 2)
            l_style += l_1
        del aux1
        del aux2
        del l_1

        torch.cuda.empty_cache()

        return l_style

    def l_tv(self):
        """
        Compute the smoothing penalty: total variation loss
        :return: total variation value
        """
        B, C, H, W = self.Icomp.shape
        Ncomp = B * C * H * W
        
        """
        l_tv = 0.0
        for b in range(B):
            for c in range(C):
                hole_pix = (self.mask[b,c,:,:] == 0).nonzero()
                first_col = self.Icomp[b,c,:,0][:,None]
                last_col = self.Icomp[b,c,:,-1][:,None]
                enhanced1 = torch.cat((first_col,self.Icomp[b,c,:,:]),dim=1)
                enhanced2 = torch.cat((self.Icomp[b,c,:,:], last_col),dim=1)
                res = enhanced1 - enhanced2

                del enhanced1
                del enhanced2
                l1_cols = torch.norm(res[hole_pix[:,0],hole_pix[:,1]], p=1)
                first_row = self.Icomp[b, c, 0, :][None,:]
                last_row = self.Icomp[b, c, -1, :][None,:]
                enhanced1 = torch.cat((first_row, self.Icomp[b,c,:,:]), dim=0)
                enhanced2 = torch.cat((self.Icomp[b,c,:,:], last_row), dim=0)
                res = enhanced1 - enhanced2
                del enhanced1
                del enhanced2
                l1_rows = torch.norm(res[hole_pix[:,0],hole_pix[:,1]], p=1)
                l_tv += (l1_cols + l1_rows)/Ncomp

        del hole_pix
        del res
        del l1_rows
        del l1_cols
        del first_col
        del first_row
        del last_col
        del last_row
        torch.cuda.empty_cache()
        """
        
        aux1 = self.Icomp[:, :, :, :-1] - self.Icomp[:, :, :, 1:]
        aux2 = self.Icomp[:, :, :-1, :] - self.Icomp[:, :, 1:, :]
        
        l1_loss = torch.norm(aux1[self.mask[:, :, :, :-1] == 1], p=1) + torch.norm(aux2[self.mask[:, :, :-1, :] == 1], p=1)
        l_tv = l1_loss/Ncomp

        del aux1, aux2, l1_loss
        torch.cuda.empty_cache()

        return l_tv

def Fit(val_set=None, learning_rate=.00005, n_epochs=10, batch_size=6, patience = 5, learning_rate_decay = None):
    print("inicio de fit...")
    print("Debut de fit...")
    print("Αρχή του fit...")
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

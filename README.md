# Image reconstruction

## Download the Large-scale CelebFaces Attributes (CelebA) Dataset from their Google Drive link :

CelebA: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
Google Drive: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8

python3 get_drive_file.py 0B7EVK8r0v71pZjFTYXZWM3FlRnM celebA.zip

## Model :

Programming Part folder : 
train.py -> train the model and save checkpoints and evolution of the losses on the training and validation sets.
test.py -> call a checkpoint of the model and compute the reconstruction on a test set and the accuracy of the reconstruction according to the 3 metrics : L1, PSNR and MSSSIM.

## Results :

### Results after training on 1000 images for 38 epochs :
![picture alt](./results/results.PNG)

### Comparisons with 2 techniques, Patch Match [1] and Glocal and Local discriminators [2] : 
![picture alt](./results/comp.PNG)

[1] Barnes, C., Shechtman, E., Finkelstein, A., Goldman, D.B.: \textit{Patchmatch: A randomized correspondence algorithm for structural image editing}. ACM Transactions on Graphics-TOG \textbf{28}(3), 24 (2009)

[2] Iizuka, S., Simo-Serra, E., Ishikawa, H.: \textit{Globally and locally consistent image completion}. ACM Transactions on Graphics (TOG) \textbf{36}(4), 107 (2017)

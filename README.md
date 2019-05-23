# Image reconstruction - KTH DD2424 - Deep Learning Project

Reproduction of [1].

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

### Comparisons with 2 techniques, Patch Match [2] and Glocal and Local discriminators [3] : 
![picture alt](./results/comp.PNG)

[1] G. Liu, F. A. Reda, K. J. Shih, T.-C. Wang, A. Tao, and B. Catanzaro, *Image Inpainting for Irregular Holes Using Partial Convolutions*, apr 2018. [Online]. Available: http://arxiv.org/abs/1804.07723

[2] Barnes, C., Shechtman, E., Finkelstein, A., Goldman, D.B.: *Patchmatch: A randomized correspondence algorithm for structural image editing*. ACM Transactions on Graphics-TOG **28**(3), 24 (2009) Website : https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/patchmatch.pdf Available : https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/index.php

[3] Iizuka, S., Simo-Serra, E., Ishikawa, H.: *Globally and locally consistent image completion*. ACM Transactions on Graphics (TOG) **36**(4), 107 (2017) Available : http://iizuka.cs.tsukuba.ac.jp/projects/completion/data/completion_sig2017.pdf

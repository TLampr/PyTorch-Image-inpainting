#squared images 

class UNet(nn.Module):
    def __init__(self, img_shape):
        super(UNet, self).__init__()
        
        #Todo : nb_jump = get_params()
        nb_jump = 3
        
        i = int(np.log2(512/img_shape))
        
        kernel_sizes = [7, 5, 5, 3, 3, 3, 3, 3]
        
        nb_layers = 8 - i
        nb_channels = []
        
        size = img_shape
        for j in range(nb_jump):
          size = int(size/2)
          nb_channels.insert(0, size)

        for j in range (nb_layers - nb_jump) :
          nb_channels.append(img_shape)
        
        # ENCODING : 
        self.encoding_list= nn.ModuleList() 
            
        self.encoding_list.append(
            nn.Sequential(
              nn.Conv2d(3, nb_channels[0], kernel_size=kernel_sizes[0], stride=2, 
                        padding=int((kernel_sizes[0] - 1)/2)),
              nn.ReLU())
              )
     
        for j in range(1, nb_layers) :
            self.encoding_list.append(
                nn.Sequential(
                  nn.Conv2d(nb_channels[j-1], nb_channels[j], 
                            kernel_size=kernel_sizes[j], stride=2,
                            padding=int((kernel_sizes[j] - 1)/2)),
                  nn.BatchNorm2d(nb_channels[j]),
                  nn.ReLU())
                  )
 
        # DECODING :
        
        #upsampling and concatenation in forward pass
           
        self.decoding_list= nn.ModuleList()
        for j in range(nb_layers-1) :
          self.decoding_list.append(
            nn.Sequential(
              nn.Conv2d(nb_channels[nb_layers-j-1]+nb_channels[nb_layers-j-2], nb_channels[nb_layers-j-2], kernel_size=3, stride=1, padding=1),
              nn.BatchNorm2d(nb_channels[nb_layers-j-2]),
              nn.LeakyReLU(0.2)
            )
          )
    
        self.decoding_list.append(
            nn.Sequential(
              nn.Conv2d(nb_channels[0]+3, 3, kernel_size=3, stride=1, padding=1))
            )
        #no batch norm or relu here -> output is the reconstructed image
           
        #print(self.encoding_list)
        #print(self.decoding_list)
  
    def forward(self, x):     
      output_feature = []
      out = x
      output_feature.append(out)
      for j in range(nb_layers) :
        out = self.encoding_list[j](out)
        output_feature.append(out)
           
      #torch.cat((first_tensor, second_tensor), dimension)
           
      for j in range(nb_layers) :
        nearestUpSample = nn.UpsamplingNearest2d(scale_factor=2)(out)    
        concat = torch.cat((output_feature[nb_layers - j - 1], nearestUpSample), dim=1)
        out = self.decoding_list[j](concat)
         
      return out

#Todo : chose the number of jumps
def get_params(img_shape) :  
  if img_shape == 512 :
    nb_jump = 3

  elif img_shape == 256 :
    nb_jump = 3

  elif img_shape == 128 :
    nb_jump = 3

  elif img_shape == 64 :
    nb_jump = 3
    
  elif img_shape == 32 :
    nb_jump = 3

  else :
    print("error in image size")
    
  return nb_jump

img512 = torch.rand(10, 3, 512, 512)
img256 = torch.rand(10, 3, 256, 256)
img128 = torch.rand(10, 3, 128, 128)
img64 = torch.rand(10, 3, 64, 64)
img32 = torch.rand(10, 3, 32, 32)

list_img = []
list_img.append(img512)
list_img.append(img256)
list_img.append(img128)
list_img.append(img64)
list_img.append(img32)

for img in list_img :
  img_shape = img.shape[2]
  nb_jump = 3
  i = int(np.log2(512/img_shape))
  
  print("starting layer : ", i)
  print("nb of jumps : ", nb_jump)
  
  nb_layers = 8 - i
  nb_channels = []

  size = img_shape
  for j in range(nb_jump):
    size = int(size/2)
    nb_channels.insert(0, size)

  for j in range (nb_layers - nb_jump) :
    nb_channels.append(img_shape)
    
  print("nb de layers encoding et decoding : ", nb_layers)
  print("check size : ", len(nb_channels))
  print("nb of channels in the network : ", nb_channels)

  model = UNet(img_shape)
  output = model(img)

  output_feature = []
  out = img
  output_feature.append(out)
  for j in range(nb_layers) :
    out = model.encoding_list[j](out)
    output_feature.append(out)
    print(out.shape)
  
  for j in range(nb_layers) :
    nearestUpSample = nn.UpsamplingNearest2d(scale_factor=2)(out)    
    concat = torch.cat((output_feature[nb_layers - j - 1], nearestUpSample), dim=1)
    out = model.decoding_list[j](concat)
    print(out.shape)
   


  

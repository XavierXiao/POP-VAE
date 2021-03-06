import numpy as np
import matplotlib.pyplot as plt

for batch_idx, (data, target) in enumerate(tr):
    break

recon_batch, umu, uvar,u,glb,parts,KM,theta = model(data.to('cuda'))
grid = F.affine_grid(theta, parts.view(-1,model.h,model.w).unsqueeze(1).size())
x = F.grid_sample(parts.view(-1,model.h,model.w).unsqueeze(1), 
                  grid, padding_mode='border').view(-1,model.num_parts,model.h,model.w)
p=torch.sum(x*KM,1)
y = torch.sum(KM,1)
y[y==0]=1
p = p/y

def draw(idx):
    fig=plt.figure()
    fig.add_subplot(2,1,1)
    plt.imshow(data[idx,0,:,:])
    fig.add_subplot(2,1,2)
    plt.imshow(recon_batch[idx,0,:,:].detach().cpu().numpy())
    fig.savefig('img'+str(idx)+'trueAndRecon.jpg')
    '''draw each part
    '''
    fig = plt.figure()
    for i in range(16):
        fig.add_subplot(4,4,i+1)
        plt.imshow(parts[idx,i,:,:].detach().cpu().numpy()) #put your numpy array here
    plt.show()
    fig.savefig('img'+str(idx)+'parts.jpg')
    '''draw kernel
    '''
    fig = plt.figure()
    for i in range(16):
        fig.add_subplot(4,4,i+1)
        plt.imshow(KM[idx,i,:,:].detach().cpu().numpy()) #put your numpy array here
    plt.show()
    fig.savefig('img'+str(idx)+'kernel.jpg')
    '''draw the part after transf
    '''
    fig = plt.figure()
    for i in range(16):
        fig.add_subplot(4,4,i+1)
        plt.imshow(x[idx,i,:,:].detach().cpu().numpy()) #put your numpy array here
    plt.show()
    
    fig=plt.figure()
    plt.imshow(p[idx,:,:].detach().cpu().numpy())

#A = self.kernelMatrix(5,self.nc,self.h,self.w,200,'gaussian')
#plt.imshow(A[0,0,:,:].detach().numpy())
def generatePart(z):
    fig = plt.figure()
    new_z = torch.tensor(z)
    interRecon = model.z2parts(new_z.view(-1,1,1)).view(-1,10,10)
    for i in range(z.shape[0]):
        plt.imshow(interRecon[i,:,:].detach().cpu().numpy())
        fig.savefig('z_'+str(i)+'.png')

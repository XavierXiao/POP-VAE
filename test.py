import torch
import numpy as np
import matplotlib.pyplot as plt

model_list = []
for i in range(10):
    model = POPVAE(h, w, h_dim, z_dim, u_dim, mb_size, nc,device, num_parts, part_h,part_w, args.transformation, M).to(device)
    model.load_state_dict(torch.load('./ModDict/n'+str(i)+'.model'))
    model_list.append(model)

te = torch.utils.data.DataLoader(dataset=mnist_te,
                                batch_size=mb_size,
                                shuffle=False,
                                drop_last=True, **kwargs)

for _, (data, target) in enumerate(te):
    break


def testPred():
    with torch.no_grad():
        comb_loss = torch.zeros(mb_size,10)
        fig = plt.figure()
        for i in range(10):
            
            recon_batch, umu, uvar, u,glb,parts,KM,theta = model_list[i](data)
            fig.add_subplot(5,2,i+1)
            plt.imshow(parts[0,0,:,:].detach().numpy()) #put your numpy array here
            recon_loss=F.binary_cross_entropy(recon_batch.squeeze().view(-1, x_dim), data.view(-1, x_dim), reduction='none').sum(1)
            comb_loss[:,i] = recon_loss.view(-1)
        _,pred = comb_loss.min(1)
        fig.savefig('parts.jpg')
        return pred,comb_loss

pred,cl = testPred()
data = data[pred!=target] 

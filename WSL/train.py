import numpy as np
import torch
import torch.nn.functional as F


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]
    loss = torch.sum((arr2-arr)**2)
    return lamda1*loss


def sparsity(arr, lamda2):
    loss = torch.sum(arr)
    return lamda2*loss



def ranking(scores, batch_size, nb_segment = 32): 
    loss = torch.tensor(0., requires_grad=True)
    
              
    for i in range(batch_size):
        maxn = torch.max(scores[int(i*nb_segment):int((i+1)*nb_segment)])
        maxa = torch.max(scores[int(i*nb_segment+batch_size*nb_segment):int((i+1)*nb_segment+batch_size*nb_segment)])
        tmp = F.relu(1.-maxa+maxn)
        loss = loss + tmp
        loss = loss + smooth(scores[int(i*nb_segment+batch_size*nb_segment):int((i+1)*nb_segment+batch_size*nb_segment)],8e-5)
        loss = loss + sparsity(scores[int(i*nb_segment+batch_size*nb_segment):int((i+1)*nb_segment+batch_size*nb_segment)], 8e-5)
    return loss / batch_size


def train(nloader, aloader, model, batch_size, optimizer, viz, device, nb_batch, nb_segment=32):
    total_loss = 0
    with torch.set_grad_enabled(True):
        model.train()

        n_iter = iter(nloader)
        a_iter = iter(aloader)
        for i in range(nb_batch):  
            ninput = next(n_iter)
            ainput = next(a_iter)
      
            ninput = ninput.view(batch_size * nb_segment,-1)
            ainput = ainput.view(batch_size * nb_segment, -1)
       
            input_ = torch.cat((ninput, ainput), 0).to(device)
           
            scores = model(input_)  # b*32  x 2048
            loss = ranking(scores, batch_size, nb_segment=nb_segment) 

            total_loss += loss.item()
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return total_loss / nb_batch

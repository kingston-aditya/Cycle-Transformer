import torch
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from trans_model import Transformer
from dataloader import data_make

def train(trans1, trans2, loader, opt_trans, mse, trans_scaler):
    
    loop = tqdm(loader, leave=True)
    device = "cpu"
    
    for idx, (img_emd, txt_emd) in enumerate(loop):
        
        img_emd = img_emd.to(device)
        txt_emd = txt_emd.to(device)
        
        with torch.cuda.amp.autocast():
            
            # transformer1
            out1 = trans1(txt_emd, img_emd)
            out1_loss = mse(out1, img_emd)
            
            out3 = trans2(out1, txt_emd)
            out3_loss = mse(out3, txt_emd)
            
            trans1_loss = out1_loss + out3_loss
            
            # transformer2
            out2 = trans2(img_emd, txt_emd)
            out2_loss = mse(out2, txt_emd)
            
            out4 = trans1(out2, txt_emd)
            out4_loss = mse(out4, txt_emd)
            
            trans2_loss = out4_loss + out2_loss
                
            # total loss
            tot_loss = (trans1_loss + trans2_loss)/2
            
        opt_trans.zero_grad()
        trans_scaler.scale(tot_loss).backward()
        trans_scaler.step(opt_trans)
        trans_scaler.update()
        
        
def main():
    
    trans1 = Transformer(512,512,512,8,3,3,4,0.3,'cpu')    
    trans2 = Transformer(512,512,512,8,3,3,4,0.3,'cpu')
    learning_rate = 3e-4
    batch_size = 32
    epochs = 1000
    
    opt_trans = optim.Adam(list(trans1.parameters()) + list(trans2.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    
    mse = nn.MSELoss()
    
    dataset = data_make(root_img="/u/home/a/asarkar/scratch/cycle_transformer/train_image.pt",       root_txt="/u/home/a/asarkar/scratch/cycle_transformer/train_text.pt",data_length = 500)
    
    print(type(dataset))
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    trans_scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(epochs):
        print("Epoch ", epoch, "out of ", epochs)
        train(trans1, trans2, loader, opt_trans, mse, trans_scaler)
        
if __name__ == "__main__":
    main()
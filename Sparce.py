# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 00:35:55 2022

@author: lucam
"""

import torch
import numpy as np
from torch import nn
from torch import optim

        
class Classification_ReadOuts:
    
    def __init__(self,N,N_class,batch_size, Pn):
        
        
        self.N=N
        
        self.N_class=N_class
                
        self.batch_size=batch_size
                
        self.Ws=[]
        
        self.theta_g=[]
        
        self.theta_i=[]
        
        self.loss=[]
        self.opt=[]
        self.opt_theta=[]
        
        self.N_copies=[]
        
    
    def Initialise_Online(self,scan,N_copies,alpha_size):
        
        
        self.N_copies=N_copies
        if scan==False:
            
            self.N_copies=1
            alpha_sizes=[alpha_size]
                               
        
        if scan==True:
            
            alpha_sizes=0.01*2**(-np.linspace(0,7,self.N_copies))

        
        self.loss = nn.BCEWithLogitsLoss()
        
        for i in range(self.N_copies):
            
            
            self.Ws.append(nn.Parameter( (2*torch.rand([self.N,self.N_class])-1)/(self.N/10)))
            self.opt.append(optim.Adam([{'params': self.Ws, 'lr':alpha_sizes[i] }]))
        
        
    
    def Online_Step(self,state,y_true):
        
       
        y=[]
        error=[]
        
        loss = nn.BCEWithLogitsLoss()
                
        for i in range(self.N_copies):
            
            
            y.append(torch.matmul(state,self.Ws[i]) )
            
            error.append(loss(y[i],y_true))
            error[i].backward()

        
            self.opt[i].step()
            self.opt[i].zero_grad()
            
            
                
        return y, error
    
    
    def Online_Evaluate(self,state,y_true):
    
        y=[]
        Acc=[]    
        error=[]            
        
        loss = nn.BCEWithLogitsLoss()
                
        for i in range(self.N_copies):
            
            
            y.append(torch.matmul(state,self.Ws[i]))
            
            error.append(loss(y[i],y_true))
            
            Acc.append(torch.mean( torch.eq(torch.argmax(y[i],dim=1),torch.argmax(y_true,dim=1)).float() ))
            
            
        return y, Acc, error
    


    def Initialise_SpaRCe(self,X_tr,alpha_size, beta, Pn):
        
        
        Pns=Pn
        
        self.N_copies=np.shape(Pns)[0]
        
        self.loss = nn.BCEWithLogitsLoss()
        
        
        for i in range(self.N_copies):
            
            
            theta_g_start=np.percentile(np.abs(X_tr),Pns[i],0)
            
            self.theta_g.append(torch.from_numpy(theta_g_start).float())
            
            
            self.theta_i.append(nn.Parameter(torch.zeros([self.N])))
            
            self.Ws.append(nn.Parameter( (2*torch.rand([self.N,self.N_class])-1)/(self.N/10) ))
            
            self.opt.append(optim.Adam([{'params': self.Ws, 'lr':alpha_size },{'params': self.theta_i, 'lr':beta }]))
            
            
    
    def SpaRCe_Step(self,state,y_true):
        
        
        state_sparse=[]
        y=[]
        error=[]
        
        loss = nn.BCEWithLogitsLoss()

        for i in range(self.N_copies):
            
            
            state_sparse.append(torch.sign(state)*torch.relu(torch.abs(state)-self.theta_g[i]-self.theta_i[i]))     
    
            y.append(torch.matmul(state_sparse[i],self.Ws[i]))
                
            error.append(loss(y[i],y_true))
            
            error[i].backward()

            self.opt[i].step()
            self.opt[i].zero_grad()
            
    
            
        return y, error, state_sparse
    
    def SpaRCe_Evaluate(self,state,y_true):
    
        state_sparse=[]
        y=[]
        Acc=[]    
        error=[]            
        sparsity=[]
        
        loss = nn.BCEWithLogitsLoss()
        
        N_cl=torch.sum(state!=0)
        
        for i in range(self.N_copies):
            
            
            state_sparse.append(torch.sign(state)*torch.relu(torch.abs(state)-self.theta_g[i]-self.theta_i[i]))     
    
            y.append(torch.matmul(state_sparse[i],self.Ws[i]))
            
            error.append(loss(y[i],y_true))
            
            Acc.append(torch.mean( torch.eq(torch.argmax(y[i],dim=1),torch.argmax(y_true,dim=1)).float() ))
            
            sparsity.append(torch.sum(state_sparse[i]!=0)/N_cl)
            
        return y, Acc, error, sparsity, state_sparse
            
    
    def Ridge_Regression(self,X_tr,Y_tr,X_te,Y_te):
    
        
        gammas=[0,0.001,0.002,0.004,0.008,0.016,0.032,0.064,0.128,0.256]
        
        
        self.N_copies=np.shape(gammas)[0]
        self.Ws=[]
        y_tr=[]
        y_te=[]
        Acc_tr=[]
        Acc_te=[]
        
        
        for i in range(self.N_copies):
            
            Y=torch.transpose(Y_tr,0,1)
            
            W=torch.matmul( torch.matmul(Y,X_tr), torch.inverse( torch.matmul(torch.transpose(X_tr,0,1),X_tr) + gammas[i]*torch.eye(self.N) ) )
            
            self.Ws.append(torch.transpose(W,0,1))
                    
            y_tr.append(torch.matmul(X_tr,self.Ws[i]))
            
            Acc_tr.append(torch.mean( torch.eq(torch.argmax(y_tr[i],dim=1),torch.argmax(Y_tr,dim=1)).float() ))
            
            y_te.append(torch.matmul(X_te,self.Ws[i]))
            
            Acc_te.append(torch.mean( torch.eq(torch.argmax(y_te[i],dim=1),torch.argmax(Y_te,dim=1)).float() ))
        
            
        
            
        return y_tr, y_te, Acc_tr, Acc_te
    


class Pruning:


    def __init__(self, W_Best, theta_g_Best, theta_i_Best, N_copies):
        
        
        self.N=W_Best.size()[0]
        
        self.N_class=W_Best.size()[1]
                
        self.N_copies=N_copies
        
        self.W_Best=W_Best
        
        self.Ws=[]
        self.theta_g=[]
        self.theta_i=[]
        
        for i in range(N_copies):         
            
            self.Ws.append(W_Best)
        
            self.theta_g.append(theta_g_Best)
        
            self.theta_i.append(theta_i_Best)
        
        
    def SpaRCe_Evaluate(self,state,y_true):
    
        state_sparse=[]
        y=[]
        Acc=[]    
        error=[]            
        sparsity=[]
        
        loss = nn.BCEWithLogitsLoss()
        
        N_cl=torch.sum(state!=0)
        
        for i in range(self.N_copies):
            
            
            state_sparse.append(torch.sign(state)*torch.relu(torch.abs(state)-self.theta_g[i]-self.theta_i[i]))     
    
            y.append(torch.matmul(state_sparse[i],self.Ws[i]))
            
            error.append(loss(y[i],y_true))
            
            Acc.append(torch.mean( torch.eq(torch.argmax(y[i],dim=1),torch.argmax(y_true,dim=1)).float() ))
            
            sparsity.append(torch.sum(state_sparse[i]!=0)/N_cl)
            
        return y, Acc, error, state_sparse
    
    
    
    def Online_Evaluate(self,state,y_true):
    
        y=[]
        Acc=[]    
        error=[]            
        
        loss = nn.BCEWithLogitsLoss()
                
        for i in range(self.N_copies):
            
            
            y.append(torch.matmul(state,self.Ws[i]))
            
            error.append(loss(y[i],y_true))
            
            Acc.append(torch.mean( torch.eq(torch.argmax(y[i],dim=1),torch.argmax(y_true,dim=1)).float() ))
            
            
        return y, Acc, error
            
    
    
    
    def Prune(self,  X_tr, Y_tr, X_te, Y_te, SpaRCe_True, N_cuts):
        
        
        
        images=torch.clone(X_tr[:,:])
        labels=torch.clone(Y_tr[:,:])
        
        
        if SpaRCe_True:
            
            Out_tr, Acc_tr, Error_tr, S=self.SpaRCe_Evaluate(images,labels)
            
            
            Active=torch.mean((S[0]!=0).float(),0)

            th=np.linspace(0.0,1,201)
            N_cuts=np.zeros(np.shape(th)[0])
            
            for i in range(np.shape(th)[0]):
                
                N_cuts[i]=torch.sum((Active>th[i])==0)
                
                
        else:
            
            Out_tr, Acc_tr, Error_tr=self.Online_Evaluate(images,labels)
            
            print('Provide the number of nodes to be deleted')
            print(N_cuts)
            
        
        Out_Sp=[]
        Acc_Sp=[]
        Err_Sp=[]
        
        Out_Rand=[]
        Acc_Rand=[]
        Err_Rand=[]
        
        Out_W=[]
        Acc_W=[]
        Err_W=[]
                
        
        for i in range(np.shape(N_cuts)[0]):
            
            
            images=torch.clone(X_te[:,:])
            labels=torch.clone(Y_te[:,:])
            
            N_cut=np.copy(N_cuts[i])
            
            
            if SpaRCe_True:
            
                Mask=Active>th[i]
                Mask=torch.unsqueeze(Mask,1).repeat(1,self.N_class).float()
                self.Ws[0]=torch.clone(nn.Parameter(self.W_Best*Mask))
            
                
                Out_te_Sp, Acc_te_Sp, Error_te_Sp, _=self.SpaRCe_Evaluate(images,labels)
                
            
                Out_Sp.append(Out_te_Sp[0].detach())
                Acc_Sp.append(Acc_te_Sp[0].detach())
                Err_Sp.append(Error_te_Sp[0].detach())
            
            
            for j in range(self.N_copies):
                
                Mask=torch.randint(0,self.N,[int(N_cut)])
                self.Ws[j]=torch.clone(nn.Parameter(self.W_Best))
                self.Ws[j][Mask,:]=0
            
            
            if SpaRCe_True:
            
                Out_te_Rand, Acc_te_Rand, Error_te_Rand, _=self.SpaRCe_Evaluate(images,labels)
            
            else:
                
                Out_te_Rand, Acc_te_Rand, Error_te_Rand=self.Online_Evaluate(images,labels)
                
            
            Out_Rand.append(Out_te_Rand)
            Acc_Rand.append(Acc_te_Rand)
            Err_Rand.append(Error_te_Rand)
            
            
            Fisher=torch.matmul((1-torch.sigmoid(torch.transpose(Out_tr[0],0,1))),X_tr)**2
            sort, indexes=torch.sort(torch.mean(Fisher,0))
            
                        
            Mask=indexes[0:int(N_cut)]
            self.Ws[0]=torch.clone(nn.Parameter(self.W_Best))
            self.Ws[0][Mask,:]=0
            
            if SpaRCe_True:
            
                Out_te_W, Acc_te_W, Error_te_W, _=self.SpaRCe_Evaluate(images,labels)
                
            else:
            
                Out_te_W, Acc_te_W, Error_te_W=self.Online_Evaluate(images,labels)
            
            
            Out_W.append(Out_te_W[0].detach())
            Acc_W.append(Acc_te_W[0].detach())
            Err_W.append(Error_te_W[0].detach())
            
            
        if SpaRCe_True:
            
            return Out_Sp, Acc_Sp, Err_Sp, Out_Rand, Acc_Rand, Err_Rand, Out_W, Acc_W, Err_W, N_cuts
        
        
        else:
        
            return Out_Rand, Acc_Rand, Err_Rand, Out_W, Acc_W, Err_W
        
        
        
        
def TI46Performance(Out, Y, Ts):
    
    N_te=np.shape(Ts)[0]
    
    ind_start=0
    ind_end=0
    p=0

    for j in range(N_te):

        ind_end=ind_end+Ts[j]

        p=p+np.float32( np.argmax( np.sum(Out[ind_start:ind_end,:],0) )==np.argmax( Y[ind_start,:] ) )/N_te
        
        ind_start=np.copy(ind_end)  
        
    return p
    
            
        
        
    
        
        
        
        
class Adam:

    def __init__(self, Params):
        
        N_dim=np.shape(Params.shape)[0] # It finds out if the parameters given are in a vector (N_dim=1) or a matrix (N_dim=2)
        
        # INITIALISATION OF THE MOMENTUMS
        if N_dim==1:
               
            self.N1=Params.shape[0]
            
            self.mt=torch.zeros([self.N1])
            self.vt=torch.zeros([self.N1])
        
        if N_dim==2:
            
            self.N1=Params.shape[0]
            self.N2=Params.shape[1]
        
            self.mt=torch.zeros([self.N1,self.N2])
            self.vt=torch.zeros([self.N1,self.N2])
        
        # HYPERPARAMETERS OF ADAM
        self.beta1=0.9
        self.beta2=0.999
        
        self.epsilon=10**(-8)
        
        # COUNTER OF THE TRAINING PROCESS
        self.counter=0
        
        
    def Compute(self,Grads):
        
        # Compute the Adam updates by following the scheme above (beginning of the notebook)
        
        self.counter=self.counter+1
        
        self.mt=self.beta1*self.mt+(1-self.beta1)*Grads
        
        self.vt=self.beta2*self.vt+(1-self.beta2)*Grads**2
        
        mt_n=self.mt/(1-self.beta1**self.counter)
        vt_n=self.vt/(1-self.beta2**self.counter)
        
        New_grads=mt_n/(torch.sqrt(vt_n)+self.epsilon)
        
        return New_grads
        
        
        
           
                
        
        
        

        
        
        
        
    
    
    
    
    
    
                   
    
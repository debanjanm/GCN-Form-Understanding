##==============================================================================================================##
import torch.utils.data as data
import glob
import time

from random import randrange
from dgl.nn.pytorch import GATConv, GraphConv
#import matplotlib.pyplot as plt
import dgl.function as fn

from torch import nn
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import json
from sklearn.metrics import adjusted_rand_score

from gensim.models import fasttext
# from gensim.models import FastText
from sentence_transformers import SentenceTransformer

import numpy as np
import xml.etree.ElementTree as ET
import networkx as nx
import pdb
# import pagexml
import re
import torch 
import dgl
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from utils import *
from evaluate import *
from model import Net
from test import *

# from sent_embed_function import sent_emb
# %matplotlib inline 
##==============================================================================================================##
# from torch_geometric.nn import GATv2Conv

##==============================================================================================================##
##S - BERT
# model_path = "C:\\Users\\acer\\OneDrive\\Documents\\GitHub\\GCN-Form-Understanding\\model\\sbert\\"
# sbert = SentenceTransformer(model_path + 'all-mpnet-base-v2')

# sbert = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

modelPath = "C:/Users/acer/OneDrive/Documents/GitHub/GCN-Form-Understanding/model/sbert-local/"
# sbert.save(modelPath)
sbert = SentenceTransformer(modelPath)

# #Our sentences we like to encode
# sentences = 'This framework generates embeddings for each input sentence'

# #Sentences are encoded by calling model.encode()
# embeddings = sbert.encode(sentences)

# embeddings.shape

##==============================================================================================================##
##Fasttext 
ft_model_path = "C:\\Users\\acer\\OneDrive\\Documents\\GitHub\\GCN-Form-Understanding\\model\\fasttext\\"
# Fasttext
ft_model = fasttext.load_facebook_model(ft_model_path+"crawl-300d-2M-subword.bin")

# ft_model_path = "C:\\Users\\1628769\\OneDrive - Standard Chartered Bank\\Documents\\aiml\\infox\\graph_engine_v0.0.1b\\entity_linking_with_gnn\\model\\fasttext\\"
# ft_model_0 = fasttext.load_facebook_model(ft_model_path+"crawl-300d-2M-subword.bin")
# ft_model_1 = fasttext.load_facebook_model(ft_model_path+"own_fasttext_model_pretrained.bin")

# def fasttext_norm_emb(model_0, model_1, text):
#     old_vector = model_0.wv[text]  # Grab the existing vector
#     new_vector = model_1.wv[text]

#     final_emb=np.log(new_vector / old_vector)

#     final_emb = np.where(
#         np.isnan(final_emb),
#         0,
#         final_emb,
#     )

#     c = 1 / (1 + np.exp(-final_emb))
#     return c

##==============================================================================================================##
##Dataset
def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs,entity_links = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    #labels = dgl.batch(labels)
    return batched_graph,entity_links#torch.tensor(labels)
    #return batched_graph, group_labels,entity_labels,entity_links#torch.tensor(labels)

class FUNSD(data.Dataset):
    """PagesDataset
    """
  
    def __init__(self, root_path, ft_model):
        self.root = root_path

        # List of files and corresponding labels
        self.files =glob.glob(root_path+'/*.json')
                    #os.listdir(root_path)
        
        self.unique_labels = ['question','answer','header','other']
        #Fasttext
        # if(type(ft_model)=='str'):
        #     self.embeddings = fasttext.load_facebook_model("model/fasttext/crawl-300d-2M-subword.bin")
        # self.embeddings = fasttext.load_facebook_model("model/fasttext/crawl-300d-2M-subword.bin")

        #S-Bert
        if(type(ft_model)=='str'):
            self.embeddings = sbert
        self.embeddings = sbert

        # #S-Level
        # if(type(ft_model)=='str'):
        #     self.embeddings = fasttext_norm_emb
        # self.embeddings = fasttext_norm_emb

    def __getitem__(self, index):
        # Read the graph and label
        G,entity_links =self.read_annotations(self.files[index])
        
        # ENSURE BIDIRECTED
        g_in=G
   
        return g_in,entity_links

    def label2class(self, label):
        # Converts the numeric label to the corresponding string
        return self.unique_labels[label]

    def class2label(self,c):
        label = self.unique_labels.index(c)
        return label

    def __len__(self):
        # Subset length
        return len(self.files)
    
    def read_annotations(self,json_file):
        # Input: json file path with page ground truth
        # Output:   - Graph to be given as input to the network
        #           - Dictionary with target edge predictions over input graph
        #           - List of entity links
        
#         Data/training_data/annotations\0060036622.json
#         0060036622
#         Data
        form_id = re.split(r"\/|\\", json_file)[-1].split('.')[0]
        partition = json_file.split('/')[-2]
        image_file = os.path.join('dataset',partition,'images',form_id+'.png')
        im = plt.imread(image_file)
        
        image_h,image_w= im.shape
        with open(json_file, errors="ignore") as f:
            data = json.load(f)
        form_data = data['form']
        
        entity_idx = 0
        entity_shapes = []
        entity_links=[]
        entity_positions=[]
        entity_embeddings =[]
        
        # Get total amount of words in the form and their attr to create am.
        for entity in form_data:
            for link in entity['linking']:
                if link not in entity_links and [link[1],link[0]] not in entity_links:
                    entity_links.append(link) 

            #FastText
            # entity_embeddings.append(self.embeddings.wv[entity['text']])
            
            #S-Bert
            entity_embeddings.append(self.embeddings.encode(entity['text']))

            #BERT-Level
            # print(f"entity['text']:{entity['text']}")
            # entity_embeddings.append(self.embeddings(entity['text'],768,0))

            #Fasttext-Level
            # print(f"entity['text']:{entity['text']}")
            # entity_embeddings.append(self.embeddings(ft_model_0, ft_model_1, entity['text']))

            # len(ft_model_0.wv['Date'])
            # sum(ft_model_0.wv['Date'])
            # len(fasttext_norm_emb(ft_model_0, ft_model_1, 'Date'))
            # a = fasttext_norm_emb(ft_model_0, ft_model_1, 'Date')
            # b = (a - np.min(a)) / np.ptp(a)
            # sum(a)
            
            # c = a / (np.linalg.norm(a) + 1e-16)
            # sum(c)

            entity_position = np.array(entity['box'][:2])
            entity_positions.append(entity_position)
           
            entity_shape  = np.array([entity['box'][2] - entity['box'][0],entity['box'][3] - entity['box'][1]])
            entity_shapes.append(entity_shape)
            
            entity_idx+=1
        

        entity_embeddings = torch.tensor(entity_embeddings)
        entity_positions = torch.tensor(entity_positions).float()          
        entity_shapes = torch.tensor(entity_shapes).float()  

        #normalize positions with respect to page
        entity_positions.float()
        entity_positions[:,1]=entity_positions[:,1]/float(image_h)
        entity_positions[:,0]=entity_positions[:,0]/float(image_w)
        entity_positions-=0.5

        entity_shapes[:,1]=entity_shapes[:,1]/float(image_h)
        entity_shapes[:,0]=entity_shapes[:,0]/float(image_w)
        
        entity_graph_nx = nx.complete_graph(len(form_data))
        entity_graph = dgl.DGLGraph()
        entity_graph = dgl.from_networkx(entity_graph_nx)
        
        entity_graph = dgl.to_bidirected(entity_graph)
        entity_graph_edges = torch.t(torch.stack([entity_graph.edges()[0],entity_graph.edges()[1]]))
        
        
        entity_graph.ndata['position']=entity_positions
        entity_graph.ndata['w_embed']=entity_embeddings
        entity_graph.ndata['shape']=entity_shapes

        entity_link_labels = []
        for edge in entity_graph_edges.tolist():
            if edge in entity_links or [edge[1],edge[0]] in entity_links:
                entity_link_labels.append(1)
            else:
                entity_link_labels.append(0)
        entity_link_labels=torch.tensor(entity_link_labels)
        
        return entity_graph, entity_link_labels

##==============================================================================================================##
##Load data
train_dir= 'dataset/training_data/annotations'
test_dir = 'dataset/testing_data/annotations'

trainset = FUNSD(train_dir,ft_model='ft_model')
validset = FUNSD(test_dir,ft_model='ft_model')
testset = FUNSD(test_dir,ft_model='ft_model')

train_loader = DataLoader(trainset, batch_size=1, shuffle=True, collate_fn=collate)
valid_loader = DataLoader(validset, batch_size=1, collate_fn=collate)
test_loader = DataLoader(testset, batch_size=1, collate_fn=collate)

##==============================================================================================================##
##Train model
EPOCH = 1
EMBED_DIM = 772
# EMBED_DIM = 304
HIDDEN_DIM = 128

# torch. __version__

epoch_losses = []

def train(model):
    if torch.cuda.is_available():
        model = model.cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, 5, gamma = 0.9)
    model.train()
    
    def random_choice(tensor,k=100):
        perm = torch.randperm(tensor.size(0))
        idx = perm[:k]
        samples = tensor[idx]
        return samples

    train_log = open('train_log.txt','w')
    train_log.close()
    best_acc =0
    best_components_error = 200
    patience = 100
    epochs_no_improvement=0

    for epoch in range(EPOCH):
        epoch_loss = 0
        epoch_link_loss = 0
        print("\n\n")
        model.training=True
        for iter, (input_graph,link_labels) in enumerate(train_loader):
            
            if torch.cuda.is_available():
                input_graph = input_graph.to(torch.device('cuda:0'))

            #for iter, (input_graph, group_labels,entity_labels,link_labels) in enumerate(train_loader):
            progress = 100*float(iter)/len(train_loader)
            progress = float("{:.2f}".format(progress))
            print('Epoch '+str(epoch)+' '+str(progress)+'%',end="\r")    
            #sys.stdout.flush()
            # Get predictions
            optimizer.zero_grad()
            entity_link_score = model(input_graph,link_labels)

            # convert target edges dict from complete graph to input graph edges 0s and 1s
    
            # Entity link loss
            entity_link_labels = link_labels[0].float()
            #print('Link labels total',entity_link_labels.sum())
            class_weights = entity_link_labels.shape[0]*torch.ones(entity_link_labels.shape)
            class_weights[entity_link_labels.bool()] /= 2*entity_link_labels.sum()
            class_weights[(1-entity_link_labels).bool()] /= 2*(1-entity_link_labels).sum()
            
            if torch.cuda.is_available():
                class_weights = class_weights.to(torch.device('cuda:0'))
                entity_link_labels = entity_link_labels.to(torch.device('cuda:0'))
 
            link_loss = F.binary_cross_entropy(entity_link_score,entity_link_labels,weight=class_weights)

            loss=link_loss#+labeling_loss+group_loss 
            loss.backward()
            entity_link_score[entity_link_score>0.5]=1.
            entity_link_score[entity_link_score<=0.5]=0.

            epoch_link_loss+=float(link_loss)
            optimizer.step()

        epoch_losses.append(epoch_link_loss)
        print('\t* Epoch '+str(epoch) +' link loss '+str(float(epoch_link_loss))+ ' lr' + str(scheduler.get_lr()[0]))
        print(" Validation \n")
        accuracies = []
        scheduler.step()
        model.training=False
        
        # VALIDATION STEP
        linking_f1 = test(test_loader,model)
        epoch_acc = linking_f1
        ### END VAL
        train_log = open('train_log.txt','a')
        train_log.write('\t Epoch '+str(epoch) +' loss '+str(float(loss)) + ' val acc' + str(epoch_acc)+'\n')
        train_log.close()


        if epoch_acc > best_acc:
            best_acc = epoch_acc
            print('new best score',epoch_acc)
            torch.save(model,'model.pt')
            epochs_no_improvement=0
        else:
            epochs_no_improvement+=1
        if epochs_no_improvement>patience:
            print('Epochs no improvement',epochs_no_improvement)
            print('Training finished')
            train_log.close()
            break
    return model


model = Net(EMBED_DIM, HIDDEN_DIM)
model = train(model)

##==============================================================================================================##
##Train model
EPOCH = 1
IN_NODE_FEATS = 304
IN_EDGE_FEATS = 1
OUT_NODE_FEATS = 128
OUT_EDGE_FEATS = 128
# HIDDEN_DIM = 128

# torch. __version__

epoch_losses = []

def train(model):
    if torch.cuda.is_available():
        model = model.cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, 5, gamma = 0.9)
    model.train()
    
    def random_choice(tensor,k=100):
        perm = torch.randperm(tensor.size(0))
        idx = perm[:k]
        samples = tensor[idx]
        return samples

    train_log = open('train_log.txt','w')
    train_log.close()
    best_acc =0
    best_components_error = 200
    patience = 100
    epochs_no_improvement=0

    for epoch in range(EPOCH):
        epoch_loss = 0
        epoch_link_loss = 0
        print("\n\n")
        model.training=True
        for iter, (input_graph,link_labels) in enumerate(train_loader):
            
            if torch.cuda.is_available():
                input_graph = input_graph.to(torch.device('cuda:0'))

            #for iter, (input_graph, group_labels,entity_labels,link_labels) in enumerate(train_loader):
            progress = 100*float(iter)/len(train_loader)
            progress = float("{:.2f}".format(progress))
            print('Epoch '+str(epoch)+' '+str(progress)+'%',end="\r")    
            #sys.stdout.flush()
            # Get predictions
            optimizer.zero_grad()
            entity_link_score = model(input_graph,link_labels)

            # convert target edges dict from complete graph to input graph edges 0s and 1s
    
            # Entity link loss
            entity_link_labels = link_labels[0].float()
            #print('Link labels total',entity_link_labels.sum())
            class_weights = entity_link_labels.shape[0]*torch.ones(entity_link_labels.shape)
            class_weights[entity_link_labels.bool()] /= 2*entity_link_labels.sum()
            class_weights[(1-entity_link_labels).bool()] /= 2*(1-entity_link_labels).sum()
            
            if torch.cuda.is_available():
                class_weights = class_weights.to(torch.device('cuda:0'))
                entity_link_labels = entity_link_labels.to(torch.device('cuda:0'))
 
            link_loss = F.binary_cross_entropy(entity_link_score,entity_link_labels,weight=class_weights)

            loss=link_loss#+labeling_loss+group_loss 
            loss.backward()
            entity_link_score[entity_link_score>0.5]=1.
            entity_link_score[entity_link_score<=0.5]=0.

            epoch_link_loss+=float(link_loss)
            optimizer.step()

        epoch_losses.append(epoch_link_loss)
        print('\t* Epoch '+str(epoch) +' link loss '+str(float(epoch_link_loss))+ ' lr' + str(scheduler.get_lr()[0]))
        print(" Validation \n")
        accuracies = []
        scheduler.step()
        model.training=False
        
        # VALIDATION STEP
        linking_f1 = test(test_loader,model)
        epoch_acc = linking_f1
        ### END VAL
        train_log = open('train_log.txt','a')
        train_log.write('\t Epoch '+str(epoch) +' loss '+str(float(loss)) + ' val acc' + str(epoch_acc)+'\n')
        train_log.close()


        if epoch_acc > best_acc:
            best_acc = epoch_acc
            print('new best score',epoch_acc)
            torch.save(model,'model.pt')
            epochs_no_improvement=0
        else:
            epochs_no_improvement+=1
        if epochs_no_improvement>patience:
            print('Epochs no improvement',epochs_no_improvement)
            print('Training finished')
            train_log.close()
            break
    return model


model = Net(EMBED_DIM, HIDDEN_DIM)
model = train(model)

##==============================================================================================================##

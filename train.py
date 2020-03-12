import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import os
import pickle
import random
import argparse
import numpy as np
from data import *
from model import Encoder,Decoder

USE_CUDA = torch.cuda.is_available()

def train(config):
    
    train_data, word2index, tag2index, intent2index = preprocessing(config.file_path,config.max_length)

    if train_data==None:
        print("Please check your data or its path")
        return
    
    encoder = Encoder(len(word2index),config.embedding_size,config.hidden_size)
    decoder = Decoder(len(tag2index),len(intent2index),len(tag2index)//3,config.hidden_size*2)
    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    #print("來到這裏了！1！")
    encoder.init_weights()
    decoder.init_weights()
    #print("來到這裏了！2！")
    loss_function_1 = nn.CrossEntropyLoss(ignore_index=0)
    loss_function_2 = nn.CrossEntropyLoss()
    enc_optim= optim.Adam(encoder.parameters(), lr=config.learning_rate)
    dec_optim = optim.Adam(decoder.parameters(),lr=config.learning_rate)
    #print("來到這裏了！3！")
    for step in range(config.step_size):
        losses=[]
        for i, batch in enumerate(getBatch(config.batch_size,train_data)):
            x,y_1,y_2 = zip(*batch)
            x = torch.cat(x)
            tag_target = torch.cat(y_1)
            intent_target = torch.cat(y_2)
            # print("來到這裏了！4！")
            x_mask = torch.cat([Variable(torch.ByteTensor(tuple(map(lambda s: s ==0, t.data)))).cuda() if USE_CUDA else Variable(torch.ByteTensor(tuple(map(lambda s: s ==0, t.data)))) for t in x]).view(config.batch_size,-1)
            y_1_mask = torch.cat([Variable(torch.ByteTensor(tuple(map(lambda s: s ==0, t.data)))).cuda() if USE_CUDA else Variable(torch.ByteTensor(tuple(map(lambda s: s ==0, t.data)))) for t in tag_target]).view(config.batch_size,-1)
            #   print("來到這裏了！5！")
            encoder.zero_grad()
            decoder.zero_grad()
            #   print("來到這裏了！6！")
            output, hidden_c = encoder(x,x_mask)
            # print("來到這裏了！7！")
            start_decode = Variable(torch.LongTensor([[word2index['<SOS>']]*config.batch_size])).cuda().transpose(1,0) if USE_CUDA else Variable(torch.LongTensor([[word2index['<SOS>']]*config.batch_size])).transpose(1,0)
            # print("來到這裏了！8！")

            tag_score, intent_score = decoder(start_decode,hidden_c,output,x_mask)
            #print("來到這裏了！9！")
            loss_1 = loss_function_1(tag_score,tag_target.view(-1))
            # print("來到這裏了！10！")
            loss_2 = loss_function_2(intent_score,intent_target)
            #print("來到這裏了！11！")
            loss = loss_1+loss_2
            losses.append(loss.data.cpu().numpy() if USE_CUDA else loss.data.numpy())
            #print("來到這裏了！12！")
            loss.backward()
           # print("來到這裏了！13！")

            torch.nn.utils.clip_grad_norm(encoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm(decoder.parameters(), 5.0)

            enc_optim.step()
            dec_optim.step()

            if i % 100==0:
                with open("result.txt", "a+") as f:
                    #print("Step",step," epoch",i," : ",np.mean(losses))
                    print(f"Step是{step},epoch是{i} ：均值为{np.mean(losses)}")
                    f.write(f"Step是{step},epoch是{i} ：均值为{np.mean(losses)}")
                    f.write("\n")
                    losses=[]
    
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    #print("來到這裏了！5！")
    torch.save(decoder.state_dict(),os.path.join(config.model_dir,'jointnlu-decoder.pkl'))
    torch.save(encoder.state_dict(),os.path.join(config.model_dir, 'jointnlu-encoder.pkl'))
    print("Train Complete!")
    
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./data/atis-2.train.w-intent.iob' ,
                        help='path of train data')
    parser.add_argument('--model_dir', type=str, default='./model/' ,
                        help='path for saving trained model')

    # Model parameters
    parser.add_argument('--max_length', type=int , default=100 ,
                        help='max sequence length')
    parser.add_argument('--embedding_size', type=int , default=75 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=75 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')

    parser.add_argument('--step_size', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    config = parser.parse_args()
    train(config)
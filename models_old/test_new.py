import torch
from torch.autograd import Variable
from collections import Counter
import pickle
import random
import os
from model import Encoder,Decoder



USE_CUDA = torch.cuda.is_available()
flatten = lambda l: [item for sublist in l for item in sublist]
def prepare_sequence(seq, to_ix):
    idxs = list(map(lambda w: to_ix[w] if w in to_ix.keys() else to_ix["<UNK>"], seq))
    tensor = Variable(torch.LongTensor(idxs)).cuda() if USE_CUDA else Variable(torch.LongTensor(idxs))
    return tensor

def preprocessing(file_path, length):
    """
    atis-2.train.w-intent.iob
    """

    processed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_006_60_64_64_1_16_16_0p01/")
    print("processed_data_path : %s" % processed_path)

    if os.path.exists(os.path.join(processed_path, "processed_train_data.pkl")):
        train_data, word2index, tag2index, intent2index = pickle.load(
            open(os.path.join(processed_path, "processed_train_data.pkl"), "rb"))
        return train_data, word2index, tag2index, intent2index

    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    try:
        train = open(file_path, "r").readlines()
        print("Successfully load data. # of set : %d " % len(train))
    except:
        print("No such file!")
        return None, None, None, None

    try:
        train = [t[:-1] for t in train]
        train = [[t.split("\t")[0].split(" "), t.split("\t")[1].split(" ")[:-1], t.split("\t")[1].split(" ")[-1]] for t
                 in train]
        train = [[t[0][1:-1], t[1][1:], t[2]] for t in train]

        seq_in, seq_out, intent = list(zip(*train))
        vocab = set(flatten(seq_in))
        slot_tag = set(flatten(seq_out))
        intent_tag = set(intent)
        print(
            "# of vocab : {vocab}, # of slot_tag : {slot_tag}, # of intent_tag : {intent_tag}".format(vocab=len(vocab),
                                                                                                      slot_tag=len(
                                                                                                          slot_tag),
                                                                                                      intent_tag=len(
                                                                                                          intent_tag)))
    except:
        print(
            "Please, check data format! It should be 'raw sentence \t BIO tag sequence intent'. The following is a sample.")
        print(
            "BOS i want to fly from baltimore to dallas round trip EOS\tO O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip atis_flight")
        return None, None, None, None

    sin = []
    sout = []

    for i in range(len(seq_in)):
        temp = seq_in[i]
        if len(temp) < length:
            temp.append('<EOS>')
            while len(temp) < length:
                temp.append('<PAD>')
        else:
            temp = temp[:length]
            temp[-1] = '<EOS>'
        sin.append(temp)

        temp = seq_out[i]
        if len(temp) < length:
            while len(temp) < length:
                temp.append('<PAD>')
        else:
            temp = temp[:length]
            temp[-1] = '<EOS>'
        sout.append(temp)

    word2index = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    for token in vocab:
        if token not in word2index.keys():
            word2index[token] = len(word2index)

    tag2index = {'<PAD>': 0}
    for tag in slot_tag:
        if tag not in tag2index.keys():
            tag2index[tag] = len(tag2index)

    intent2index = {}
    for ii in intent_tag:
        if ii not in intent2index.keys():
            intent2index[ii] = len(intent2index)

    train = list(zip(sin, sout, intent))

    train_data = []

    for tr in train:
        temp = prepare_sequence(tr[0], word2index)
        temp = temp.view(1, -1)

        temp2 = prepare_sequence(tr[1], tag2index)
        temp2 = temp2.view(1, -1)

        temp3 = Variable(torch.LongTensor([intent2index[tr[2]]])).cuda() if USE_CUDA else Variable(
            torch.LongTensor([intent2index[tr[2]]]))

        train_data.append((temp, temp2, temp3))

    pickle.dump((train_data, word2index, tag2index, intent2index),
                open(os.path.join(processed_path, "processed_train_data.pkl"), "wb"))
    pickle
    print("Preprocessing complete!")

    return train_data, word2index, tag2index, intent2index





_,word2index,tag2index,intent2index = preprocessing('../dataset/corpus/atis-2.train.w-intent.iob',60)

index2tag = {v:k for k,v in tag2index.items()}
index2intent = {v:k for k,v in intent2index.items()}

encoder = Encoder(len(word2index),64,64)
decoder = Decoder(len(tag2index),len(intent2index),len(tag2index)//3,64*2)

encoder.load_state_dict(torch.load('models_006_60_64_64_1_16_16_0p01/jointnlu-encoder.pkl'))
decoder.load_state_dict(torch.load('models_006_60_64_64_1_16_16_0p01/jointnlu-decoder.pkl'))
if USE_CUDA:
    encoder = encoder.cuda()
    decoder = decoder.cuda()


test = open("./dataset/corpus/atis.test.w-intent.iob","r").readlines()
test = [t[:-1] for t in test]
test = [[t.split("\t")[0].split(" "),t.split("\t")[1].split(" ")[:-1],t.split("\t")[1].split(" ")[-1]] for t in test]
test = [[t[0][1:-1],t[1][1:],t[2]] for t in test]

#index = random.choice(range(len(test)))
error=0
with open("result.txt", 'a+') as f:
    for index in range(len(test)):

        test_raw = test[index][0]
        test_in = prepare_sequence(test_raw,word2index)
        test_mask = Variable(torch.ByteTensor(tuple(map(lambda s: s ==0, test_in.data)))).cuda() if USE_CUDA else Variable(torch.ByteTensor(tuple(map(lambda s: s ==0, test_in.data)))).view(1,-1)
        start_decode = Variable(torch.LongTensor([[word2index['<SOS>']]*1])).cuda().transpose(1,0) if USE_CUDA else Variable(torch.LongTensor([[word2index['<SOS>']]*1])).transpose(1,0)

        output, hidden_c = encoder(test_in.unsqueeze(0),test_mask.unsqueeze(0))
        tag_score, intent_score = decoder(start_decode,hidden_c,output,test_mask)

        v,i = torch.max(tag_score,1)
        #print("Input Sentence : ",*test[index][0])
        #print("Truth        : ",*test[index][1])
        #print("Prediction : ",*list(map(lambda ii:index2tag[ii],i.data.tolist())))
        v,i = torch.max(intent_score,1)
        if (test[index][2]) != (index2intent[i.data.tolist()[0]]):
            print(f"第{index}错误")
            error = error + 1
            str =" ".join(test[index][0])
            print("分类错误的第 %d 句：%s"%(error,str), file=f)
            f.write(f"Truth        : {test[index][2]}")
            f.write("\n")
            f.write(f"Prediction :   {index2intent[i.data.tolist()[0]]}")

            f.write("\n")
            f.write("\n")
    f.write(f"总计有{len(test)}个句子，其中错误分类的句子有{error}个")
    f.write("\n")
    f.write(f"当前的错误率：{(error / len(test)) * 100}%")
    f.write("\n")
    f.write(f"目标错误率是：2.35%")


print(f"总计有{len(test)}个句子，其中错误分类的句子有{error }个")
print(f"当前的错误率：{(error/len(test))*100}%")
print(f"目标错误率是：2.35%")


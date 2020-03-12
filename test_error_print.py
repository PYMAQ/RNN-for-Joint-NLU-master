from data import *
from model import Encoder,Decoder

_,word2index,tag2index,intent2index = preprocessing('../dataset/corpus/atis-2.train.w-intent.iob',60)

index2tag = {v:k for k,v in tag2index.items()}
index2intent = {v:k for k,v in intent2index.items()}

encoder = Encoder(len(word2index),75,75)
decoder = Decoder(len(tag2index),len(intent2index),len(tag2index)//3,75*2)

encoder.load_state_dict(torch.load('model/jointnlu-encoder.pkl'))
decoder.load_state_dict(torch.load('model/jointnlu-decoder.pkl'))
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


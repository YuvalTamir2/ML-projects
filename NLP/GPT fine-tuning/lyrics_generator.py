import torch
import torchtext
import spacy
import os
from torchtext.data import Field, BucketIterator, TabularDataset
from torch.utils.data import Dataset, DataLoader
import transformers
import pandas as pd
import random

project_path = r'.'

###new method - pretrained NLP models, and fine-tuning them : 

tokenizer2 = transformers.GPT2Tokenizer.from_pretrained('gpt2')
model = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
initial_states = model.state_dict()
initial_ref = initial_states['lm_head.weight'][0][-1].item()

def read_song(song):
    song_lines = []
    with open(song,'r') as song:
        for line in song:
            if len(line) > 3:
                line = line.strip('\n')
                song_lines.append(line)
    return song_lines

def create_df_for_pretrained(path_to_rappers = project_path, mc = 'Nas'):
    all_songs = []
    list_of_rappers = os.listdir(path_to_rappers)
    assert (mc in list_of_rappers), 'The MC you wanted is not in the database'
    for rapper in list_of_rappers:
        if rapper == mc:
            songs_list = os.listdir(os.path.join(path_to_rappers, rapper))
            for song in songs_list:
                skit = read_song(os.path.join(path_to_rappers, rapper,song))
                all_songs.append(skit)
    flat_list = [item for sublist in all_songs for item in sublist]
    return pd.DataFrame(flat_list)


songs_df =  create_df_for_pretrained(project_path, 'Nas') 
songs_df.to_csv('for_pretrained.csv')          
            
    

class SongDataSet(Dataset):
    def __init__(self, data_path =r'for_pretrained.csv'):
        super().__init__()
        
        self.lines_list = []
        
        data = pd.read_csv(data_path)
        lines = data.iloc[:,1]
        for line in lines:
            if len(str(line)) > 3:
              #  line = str(line)+" <|endoftext|>"
                self.lines_list.append(line)
            
    def __len__(self):
        return len(self.lines_list)
    
    def __getitem__(self,index):
        return self.lines_list[index]

song_dataset = SongDataSet()
dataloader = DataLoader(dataset = song_dataset, batch_size = 2, shuffle = True)

model.train()
optimizer = transformers.AdamW(model.parameters(), lr=1e-4)
MAX_LEN = 50
bla = 0
tmp_tokens = None
for epoch in range(5):
    epoch_loss = 0
    for idx,line in enumerate(dataloader):
        tokens = tokenizer2.encode(line[0], return_tensors='pt')
        if tokens.shape[-1] > MAX_LEN:
            continue
        if not torch.is_tensor(tmp_tokens):
            tmp_tokens = tokens
            continue
        else:
            if tmp_tokens.shape[-1] + tokens.shape[-1] > MAX_LEN:
                input_lines = tmp_tokens
                tmp_tokens = tokens
            else:
                tmp_tokens = torch.cat([tmp_tokens, tokens[:,1:]], dim=1)
                continue
        print('training now')
        outs = model(input_lines, labels = input_lines)
        loss,logits = outs[:2]
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        updated = model.state_dict()
        updated_alt = updated['lm_head.weight'][0][-1]
        #if updated_alt == initial_ref:
            #print('it wont change..')
        #else:
            #print('updating some stuff')
    print('epoch {} loss {}'.format(epoch, epoch_loss))

    
    
def pred_from_logits(logits):
    top_preds = logits.topk(5,dim =0)[1]
    pred = random.choice(top_preds)
    return pred
        
def generate_txt(model = model,inp_str = 'new york new york', max_len = 100):
   tokens = tokenizer2.encode(inp_str, return_tensors='pt')
   model.eval()
   with torch.no_grad():
       for i in range(max_len):
           outputs = model(tokens, labels = tokens)
           loss,logits = outputs[:2]
           softmax_logits = torch.softmax(logits[0,-1], dim = 0)
           predicted_idx = pred_from_logits(softmax_logits)
           tokens = torch.cat((tokens,predicted_idx.view(1,-1)), dim = 1)

       out_list = tokens.tolist()
       preds = tokenizer2.decode(out_list[0])
       return preds       

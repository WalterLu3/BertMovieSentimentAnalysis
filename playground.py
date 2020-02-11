import pandas as pd
import numpy as np
import torch
import transformers as ppb
from sklearn.model_selection import train_test_split
import warnings
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
warnings.filterwarnings('ignore')
def testBert(x) :     
    device = "cuda:0"
    with open('train.tsv') as file :
        train_read = pd.read_csv(file, sep = '\t')
    with open('test.tsv') as file : 
        test_read = pd.read_csv(file, sep = '\t')
    ## train_read = train_read.iloc[:10000,:]


    ##if you want to use bert model, change distilbert to bert
    
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertForSequenceClassification, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)  ##load tokenizer
    model = model_class.from_pretrained(pretrained_weights, num_labels = 5) ## load model
    model.to(device)
    print(2)
    
    ## tokenize data
    
    tokenized = train_read.iloc[:,2].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    labels = train_read.iloc[:,3]
    print(3)
    
    ## padding
    
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    print(4)



    ##masking
   
    attention_mask = np.where(padded != 0, 1, 0)
    attention_mask.shape
    

    ## Validation Split

    # Use 90% for training and 10% for validation.
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(padded, labels.tolist(),random_state=2018, test_size=0.1)
    
    # Do the same for the masks.
    
    train_masks, validation_masks, _, _ = train_test_split(attention_mask, labels.tolist(), random_state=2018, test_size=0.1)
    
    # return train_inputs, validation_inputs, train_labels, validation_labels
    ##convert Data into Tensor
    
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)
    print(5)
       
    
    ## Convert data into Data Loader for Batch Split

    batch_size = 32

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    return train_dataloader , validation_dataloader
    ##with torch.no_grad() : 
    ##    last_hidden_states = model(input_ids.to(device), attention_mask = attention_mask.to(device))


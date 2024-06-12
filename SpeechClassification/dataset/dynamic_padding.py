
import transformers
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer

def collate_batch(batch): 
    texts, labels = zip(*batch)
  
    tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')

    tokenized_texts = [tokenizer.encode_plus(
        text, 
        add_special_tokens=True, 
        max_length=512,  
        truncation=True,
        padding=False,  #padding will be applied dynamically
        return_tensors='pt'  
    ) for text in texts]
    
    input_ids = [x['input_ids'].squeeze(0) for x in tokenized_texts]  #remove extra dimensions
    attention_masks = [x['attention_mask'].squeeze(0) for x in tokenized_texts]

    #pad the sequences for this batch
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    
    return padded_input_ids, padded_attention_masks, labels


import transformers
import torch
from transformers import AutoTokenizer

#dynamic padding at dataloader level to reduce the number of unnecessary pad tokens processed
def collate_batch(batch): 
    texts, labels = zip(*batch)
    
    model_id = 'NousResearch/Llama-2-7b-hf'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    tokenizer.pad_token = tokenizer.eos_token
    
    tokenized_texts = tokenizer(
        list(texts), 
        add_special_tokens=True, 
        max_length=512,  #max input tokens for Llama 2 is 4096
        truncation=True, 
        padding='longest',  #pad all sequences in the batch to the maximum length of the batch
        #padding_side = "right",  #padding must be added at the end of the sequence
        return_tensors='pt' 
    )
    
    input_ids = tokenized_texts['input_ids']
    attention_masks = tokenized_texts['attention_mask']
    labels = torch.tensor(labels)
    
    return input_ids, attention_masks, labels

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchvision import transforms
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import classification_report
import torchmetrics
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from sklearn.metrics import auc, precision_recall_curve

def bnb_config(load_in_4bit=False):
    if load_in_4bit:
        config = BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False)
    else:
        config= BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf4",
        bnb_8bit_compute_dtype=torch.bfloat16
    )
    return config


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

class LLamaForSequenceClf(pl.LightningModule):
    def __init__(self, warmup_steps=None):
        super(LLamaForSequenceClf, self).__init__()

        self.model_id = 'NousResearch/Llama-2-7b-hf'
        
        self.base_learning_rate = 0.00002
        
        quant_config = bnb_config()
        
        self.warmup_steps = warmup_steps
        
        self.llama = AutoModelForSequenceClassification.from_pretrained(self.model_id, num_labels=1, quantization_config=quant_config, use_cache=False)
        #eos token
        self.llama.config.pad_token_id = self.llama.config.eos_token_id

        print(self.llama.config.pad_token_id)
        self.llama.gradient_checkpointing_enable() #to reduce memory
        self.llama = prepare_model_for_kbit_training(self.llama)
    
        #loRA (Low-Rank Adaptation) 

        #Comment llora in order to training only emeddings
        self.peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=12, lora_alpha=32, lora_dropout=0.1)
        self.llama = get_peft_model(self.llama, self.peft_config) 
        
        self.train_embeddings_only = True
        
        if self.train_embeddings_only:
            for param in self.llama.parameters():
                param.requires_grad = False
            #unfreeze embedding layer parameters to allow training just the embeddings
            self.llama.base_model.embeddings.requires_grad = True      
        else:
            for param in self.parameters():
                if param.data.dtype.is_floating_point:
                    param.requires_grad = True
            
            
        print_trainable_parameters(self)
        
        #criterion
        self.criterion = torch.nn.BCEWithLogitsLoss() 
        #torch metrics
        self.train_accuracy = torchmetrics.Accuracy(task='binary') 
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.test_accuracy = torchmetrics.Accuracy(task='binary')
        
        self.train_precision = torchmetrics.Precision(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary' )
        self.test_precision = torchmetrics.Precision(task='binary' )
        
        self.train_recall = torchmetrics.Recall(task='binary' )
        self.val_recall = torchmetrics.Recall(task='binary' )
        self.test_recall = torchmetrics.Recall(task='binary' )
        
        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.test_f1 = torchmetrics.F1Score(task="binary")
        
        self.train_auc = torchmetrics.AUROC(task='binary')
        self.val_auc = torchmetrics.AUROC(task='binary') 
        self.test_auc = torchmetrics.AUROC(task='binary')
    
        
    def forward(self, input_ids, attention_mask=None):

        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        return logits.squeeze(-1)
    
    
    def training_step(self, batch, batch_idx):
        
        input_ids, attention_masks, labels = batch
    
        logits = self(input_ids, attention_masks) 

        loss = self.criterion(logits, labels.float())
        
        probabilities = torch.sigmoid(logits) 
        
        predicted_classes = (probabilities >= 0.5).int()

        labels_np = labels.cpu().numpy()
        probabilities_np = probabilities.detach().cpu().numpy()
        
        precision_curve, recall_curve, _ = precision_recall_curve(labels_np, probabilities_np)
        pr_auc = auc(recall_curve, precision_curve)
        
        self.log("train_precision_recall_auc", pr_auc)   
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)      
        self.log("train_acc_micro", self.train_accuracy(predicted_classes, labels), on_step=True, on_epoch=True, prog_bar=True, logger=True)            
        self.log("train_f1", self.train_f1(predicted_classes, labels), on_epoch=True, prog_bar=True)       
        self.log("train_precision", self.train_precision(predicted_classes, labels), on_epoch=True, prog_bar=True)
        self.log("train_recall", self.train_recall(predicted_classes, labels), on_epoch=True, prog_bar=True)
        self.log("train_auc", self.train_auc(probabilities, labels.int()), on_epoch=True, prog_bar=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        
        input_ids, attention_masks, labels = batch
    
        logits = self(input_ids, attention_masks) 

        loss = self.criterion(logits, labels.float())
        
        probabilities = torch.sigmoid(logits) 
        predicted_classes = (probabilities >= 0.5).int()
        
        labels_np = labels.cpu().numpy()
        preds_np = predicted_classes.cpu().numpy()
        probabilities_np = probabilities.detach().cpu().numpy()
        
        precision_curve, recall_curve, _ = precision_recall_curve(labels_np, probabilities_np)
        pr_auc = auc(recall_curve, precision_curve)
        self.log("val_precision_recall_auc", pr_auc)
        
       
        print("VALIDATION CLASSIFICATION REPORT")
        print(classification_report(labels_np, preds_np))
        
        conf_matrix = torchmetrics.functional.confusion_matrix(predicted_classes, labels, task='binary')
        print("Confusion Matrix:\n", conf_matrix.cpu().numpy())

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.val_accuracy(predicted_classes, labels), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1", self.val_f1(predicted_classes, labels), on_epoch=True, prog_bar=True)
        self.log("val_precision", self.val_precision(predicted_classes, labels), on_epoch=True, prog_bar=True)
        self.log("val_recall", self.val_recall(predicted_classes, labels), on_epoch=True, prog_bar=True)
        self.log("val_auc", self.val_auc(probabilities, labels.int()), on_epoch=True, prog_bar=True)

        return loss
    
    
    def test_step(self, batch, batch_idx):
        
        input_ids, attention_masks, labels = batch
        logits = self(input_ids, attention_masks) 
        loss = self.criterion(logits, labels.float())
        
        probabilities = torch.sigmoid(logits) 
        predicted_classes = (probabilities >= 0.5).int()

        labels_np = labels.cpu().numpy()
        preds_np = predicted_classes.cpu().numpy()
        probabilities_np = probabilities.detach().cpu().numpy()
        
        precision_curve, recall_curve, _ = precision_recall_curve(labels_np, probabilities_np)
        pr_auc = auc(recall_curve, precision_curve)
        self.log("test_precision_recall_auc", pr_auc)
        
        print("TEST CLASSIFICATION REPORT")
        print(classification_report(labels_np, preds_np))
        
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_accd", self.test_accuracy(predicted_classes, labels), on_step=True, on_epoch=True, prog_bar=True, logger=True)  
        self.log("test_f1", self.test_f1(predicted_classes, labels), on_epoch=True, prog_bar=True)      
        self.log("test_precision", self.test_precision(predicted_classes, labels), on_epoch=True, prog_bar=True)
        self.log("test_recall", self.test_recall(predicted_classes, labels), on_epoch=True, prog_bar=True)      
        self.log("test_auc", self.test_auc(probabilities, labels.int()), on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.00002, weight_decay = 0.00005)

        reduce_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True) 
        
        return {"optimizer": optimizer, "lr_scheduler": reduce_scheduler, "monitor": "val_loss"} 
    
    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_closure, 
        on_tpu=False, using_native_amp=False, using_lbfgs=False):

        optimizer.step(closure=optimizer_closure)

        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.base_learning_rate 
                
        optimizer.zero_grad()

                
            

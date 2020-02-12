import playground, model_config,training
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time, datetime

device = 'cuda:0'
##get data input batch size
train_dataloader, validation_dataloader = playground.testBert(100)

##load model
model = model_config.model_set_parameter(num_label = 5, model_weights = 'distilbert-base-uncased')

model.to(device)
##load optimizer
optimizer = model_config.optimizer_set_parameter(model = model)

# Number of training epochs (authors recommend between 2 and 4)
epochs = 4

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
model = training.training_bert(model,seed=42,epochNum = epochs,train_dataloader=train_dataloader,validation_dataloader = validation_dataloader,scheduler = scheduler,optimizer = optimizer)

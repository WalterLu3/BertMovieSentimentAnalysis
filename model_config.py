from transformers import DistilBertForSequenceClassification, AdamW, BertConfig

## input number of labels, output a model before passing to GPU. So you need to send the model to GPU after.
## number of classes
## model weights = 'distilbert-base-uncased'
def model_set_parameter(num_label,model_weights) : 
    # if you want to change to Bert, make sure the distilbertsequenceclassification is also changed to bertsequenceclassification
    model = DistilBertForSequenceClassification.from_pretrained(
        model_weights,
        num_labels = num_label,
        output_attentions = False, # True if you want model returns attentions weights
        output_hidden_states = False, # True if you want model to return all hidden states
        )
    return model



## input model, learning rate, and epsilon 
def optimizer_set_parameter( model, learning_rate = 2e-5, epsilon = 1e-8) : 
    #The epsilon parameter eps = 1e-8 is “a very small number to prevent any division by zero in the implementation
    optimizer = AdamW(model.parameters(),lr = learning_rate, eps = epsilon)
    return optimizer







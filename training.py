import random
import numpy as np
import torch
from helper import flat_accuracy, format_time
import numpy as np
import time, datetime

#input random seed to reproduce same training. input epoch num
#output fine-tuned model

def training_bert(train_model,seed, epochNum,train_dataloader,validation_dataloader, scheduler, optimizer):
    seed_val = seed
    device = "cuda:0"
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    model = train_model
    # Store the average loss after each epoch
    epochs = epochNum
    loss_values = []
    
    # For each epoch ...
    # start to train
    for epoch_i in range(epochNum):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # record time
        t0 = time.time()

        #put model into training mode

        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader) :

            #progress update every 400 batches

            if step % 400 == 0 and not step == 0:
                    
                elapsed = format_time(time.time()-t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed)) 

            # unpack training batch
            # 'batch' content
            # [0]: input ids
            # [1]: attention mask
            # [2]: labels

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            #clear previously calculated gradient(this is useful in RNN models, however, unuseful in our case)
            model.zero_grad()

            # forward pass : evaluate model loss on this training batch
            outputs = model(input_ids = b_input_ids, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)

            # loss is stored in output[0]
            loss = outputs[0]

            total_loss += loss.item()

            #backward pass to calculate gradient
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:

            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():

                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have
                # not provided labels.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(b_input_ids,
                                attention_mask=b_input_mask)

            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1

        # Report the final accuracy for this validation run.
        print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    return model

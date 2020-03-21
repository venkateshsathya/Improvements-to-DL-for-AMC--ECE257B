import torch
import torch.optim as optim
import numpy as np
import os
import pickle
import datetime
def calculate_loss(model, data, label, batch_size, computing_device, criterion):
    n_samples = data.shape[0]
    n_minibatch = int((n_samples+batch_size-1)/batch_size)
    loss = 0
    I = np.arange(n_samples)
    for i in range(n_minibatch):
        idx = I[batch_size*i:min(batch_size*(i+1), n_samples)]
        dt = data[idx].to(computing_device)
        lbl = label[idx].to(computing_device)
        outputs = model(dt)
        l = criterion(outputs, lbl.long())
        loss += l.item()
    return loss/n_minibatch
        
def calculate_accuracy(model, data, label, batch_size, computing_device):
    n_samples = data.shape[0]
    n_minibatch = int((n_samples+batch_size-1)/batch_size)
    accuracy = 0
    I = np.arange(n_samples)
    for i in range(n_minibatch):
        idx = I[batch_size*i:min(batch_size*(i+1), n_samples)]
        dt = data[idx].to(computing_device)
        lbl = label[idx].numpy()
        output = model(dt).detach()
        output = output.cpu().numpy()
        output = np.argmax(output,axis=1)
        accuracy += np.sum(output == lbl)
    return accuracy/n_samples

def train_model(model, model_file, train_set, val_set, num_epochs, batch_size, learning_rate, criterion, computing_device):
    # Convert model to computing device
    model = model.to(computing_device)
    print("Model on CUDA?", next(model.parameters()).is_cuda)

    # Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Prepare Data
    train_data, train_labels = train_set['data'], train_set['labels']
    val_data, val_labels = val_set['data'], val_set['labels']
    n_samples = train_data.shape[0]
    n_minibatch = int((n_samples+batch_size-1)/batch_size)
    

    # Check for existing model
    loss_file = model_file.strip('.pt') + '_loss.pkl'
    accuracy_file = model_file.strip('.pt') + '_accuracy.pkl'
    if os.path.isfile(model_file):
        model.load_state_dict(torch.load(model_file))
        print('Existing model loaded.')
        # Load Loss and Accuracy
        with open(loss_file, 'rb') as handle:
            Loss = pickle.load(handle)
        with open(accuracy_file, 'rb') as handle:
            Accuracy = pickle.load(handle)
        n_prev_epochs = len(Loss['train'])
    else:
        n_prev_epochs = 0
        Loss = {}
        Accuracy = {}
        Loss['train'] = []
        Loss['valid'] = []
        Accuracy['train'] = []
        Accuracy['valid'] = []
    
    
    # Prepare early stopping
    prev_val = calculate_loss(model, val_data, val_labels, 1000, computing_device, criterion)
    stop_con = 0
    epoch = n_prev_epochs
    # Begin training procedure
    for epoch in range(num_epochs-n_prev_epochs):
        # Shuffle indices
        shuffled_idx = np.random.permutation(range(n_samples))
        model.train()
        train_loss = 0
        print('Epoch number: ', epoch, ' starting')
        print(datetime.datetime.now())
        for i in range(n_minibatch):
            idx = shuffled_idx[batch_size*i:min(batch_size*(i+1), n_samples)]
            data = train_data[idx].to(computing_device)
            labels = train_labels[idx].to(computing_device)
            
            # Zero out the stored gradient (buffer) from the previous iteration
            optimizer.zero_grad()
            
            # Perform the forward pass through the network and compute the loss
            outputs = model(data)
            loss = criterion(outputs, labels.long())
            #loss = criterion(outputs, labels.float())
            
            # Compute the gradients and backpropagate the loss through the network
            loss.backward()
            
            # Update the weights
            optimizer.step()
            
            train_loss += loss.item()
            torch.cuda.empty_cache()
            if (i%1000) == 0:
                print('Epoch Number: ', epoch, ' and Batch number: ', i, ' complete')
                print(datetime.datetime.now())
        
        model.eval()
        
        train_loss = calculate_loss(model, train_data, train_labels, 1000, computing_device, criterion)
        val_loss = calculate_loss(model, val_data, val_labels, 1000, computing_device, criterion)
        Loss['train'].append(train_loss)
        Loss['valid'].append(val_loss)
        
        train_acc = calculate_accuracy(model, train_data, train_labels, 1000, computing_device)
        val_acc = calculate_accuracy(model, val_data, val_labels, 1000, computing_device)
        Accuracy['train'].append(train_acc)
        Accuracy['valid'].append(val_acc)
        
        # Save loss and accuracy
        with open(loss_file, 'wb') as handle:
            pickle.dump(Loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(accuracy_file, 'wb') as handle:
            pickle.dump(Accuracy, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save model
        if (prev_val > val_loss):
            torch.save(model.state_dict(), model_file)
            prev_val = val_loss
            stop_con = 0
        else:
            print("Validation loss increased from %f to %f." %(prev_val, val_loss))
            model.load_state_dict(torch.load(model_file))
            prev_val = val_loss
            stop_con += 1
            if (stop_con >= 3):
                break
                
        
            
        print("Epoch %d : Training Loss = %f,  Validation Loss = %f" % (n_prev_epochs+epoch+1, train_loss, val_loss))
    print("Training complete after", epoch +1, "epochs")
    torch.save(model.state_dict(), model_file)
    
    return model, Loss, Accuracy


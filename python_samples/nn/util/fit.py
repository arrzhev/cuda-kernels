import time
import torch

def train(model, optimizer, criterion, train_loader, device):   
    model.train()
    loss_mean = 0.
    accuracy = 0.

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
            
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        loss_mean += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        accuracy += pred.eq(target.data.view_as(pred)).cpu().sum()

    loss_mean /= len(train_loader)
    accuracy /= len(train_loader.dataset)

    return loss_mean, accuracy

def validate(model, criterion, val_loader, device):
    model.eval()
    loss_mean = 0.
    accuracy = 0.

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            loss_mean += criterion(output, target).item()

            pred = output.data.max(1, keepdim=True)[1]
            accuracy += pred.eq(target.data.view_as(pred)).cpu().sum()
        
    loss_mean /= len(val_loader)
    accuracy /= len(val_loader.dataset)

    return loss_mean, accuracy

def fit(epochs, model, optimizer, criterion, scheduler, train_loader, val_loader, save_func, verbose, device):
    best_val_loss = float('inf')
    history = {'train_loss':[],'train_accuracy':[],'val_loss':[],'val_accuracy':[], 'elapsed_time':[]}
    for n in range(epochs):
        start_time = time.perf_counter()
        train_loss, train_accuracy = train(model, optimizer, criterion, train_loader, device)
        val_loss, val_accuracy = validate(model, criterion, val_loader, device)
        elapsed_time = time.perf_counter() - start_time

        current_lr = optimizer.param_groups[0]['lr']

        if verbose > 0:
            print(f"""Train Epoch {n}: Elapsed Time = {elapsed_time:.4f} LR = {current_lr:g}
                Train Loss = {train_loss:.4f} Train Accuracy = {(100. * train_accuracy):.4f}
                Validation Loss = {val_loss:.4f} Validation Accuracy = {(100. * val_accuracy):.4f}\n""")

        if save_func and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_func(model.state_dict())

        scheduler.step()
        history['elapsed_time'].append(elapsed_time)
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

    return history

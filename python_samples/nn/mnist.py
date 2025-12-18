import numpy as np
import torch
from torchvision import datasets, transforms

from util.fit import fit
from util.deterministic import apply_deterministic
import models

PATH_TO_DATA = "./.data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKERS = 4
SEED = 42
DETERMINISTIC = True

DTYPE = torch.float32
USE_EXTENSION = True

image_transform = transforms.Compose([
    transforms.ToTensor(),
    # Normalize input image following MNIST mean and standard deviation.
    transforms.Normalize((0.1307,), (0.3081,)),
    # Transform to needed dtype
    transforms.ConvertImageDtype(DTYPE),
    ])

def train_model(model, batch_size=32, epochs=25, lr=0.001, verbose=1):
    g, loader_worker_init = apply_deterministic(DETERMINISTIC, SEED)

    train_dataset = datasets.MNIST(PATH_TO_DATA, train=True, download=True, transform=image_transform)

    val_dataset = datasets.MNIST(PATH_TO_DATA, train=False, transform=image_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True,
                                                num_workers=WORKERS, worker_init_fn=loader_worker_init, generator=g)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = True,
                                            num_workers=WORKERS, worker_init_fn=loader_worker_init, generator=g)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    criterion = torch.nn.NLLLoss()

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    model.to(DEVICE)
    criterion.to(DEVICE)

    # save_func = lambda dict: torch.save(dict, f'model{i}.pth')
    save_func = None

    history = fit(epochs=epochs, model=model, optimizer=optimizer, criterion=criterion,
                    scheduler=scheduler, train_loader=train_loader, val_loader=val_loader,
                    save_func=save_func, verbose=verbose, device=DEVICE)

    print(f"Results: total time = {sum(history['elapsed_time']):.4f} val_loss = {min(history['val_loss'])}, val_acc = {history['val_accuracy'][np.argmin(history['val_loss'])]}")
    
if __name__ == '__main__':
    input_features = 28*28
    output_features = 10
    model = models.CustomMLP(input_features, output_features, USE_EXTENSION, DTYPE)
    train_model(model, batch_size=64, epochs=2, lr=0.001, verbose=1)
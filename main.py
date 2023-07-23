import warnings
warnings.filterwarnings("ignore")

import torch, torchvision, os, time, tqdm, argparse, random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, models, transforms
from loader import recaptcha_loader

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def parse_option():
    parser = argparse.ArgumentParser('ResNet train', add_help=False)
    parser.add_argument('--dataset_dir', type=str, help="dataset dir")
    parser.add_argument('--batch_size', type=int, help="batch size for single GPU")
    parser.add_argument('--epoch', type=int, help="training epoches")
    parser.add_argument('--lr', type=float, help="learning rate")
    parser.add_argument('--output', type=str, help="Run testdataset and eval")
    parser.add_argument('--resume', help='resume from checkpoint', default = None)

    args, unparsed = parser.parse_known_args()
    return args

def train_model(model, device, train_loader, valid_loader, weights_path, epochs=25, lr = 0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr= lr, momentum=0.9)

    model = model.to(device)
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )

        if (epoch + 1) % 10 == 0 :
            torch.save(model, weights_path + f'/ResNet_{epoch + 1}.pth')
    
    return model


if __name__ == "__main__" :
    # Set Hyperparameters
    args = parse_option()

    dataset_dir = args.dataset_dir
    batch_size = args.batch_size
    epochs = args.epoch
    lr = args.lr
    output_name = args.output
    checkpoint = args.resume
    
    input_size = 224
    num_classes = 10

    train_loader, val_loader = recaptcha_loader(dataset_dir, input_size, batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if checkpoint is not None :
        model = torch.load(checkpoint).to(device)
    else :
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model = model.to(device)
    train_model(model = model, device = device, train_loader=train_loader, valid_loader=val_loader, weights_path =output_name, epochs=epochs, lr=lr)
    torch.save(train_model, output_name + '/final_model.pth')
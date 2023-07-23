import torch
import numpy as np
from torchvision import datasets, transforms

def recaptcha_loader(dataset_dir, input_size, batch_size) :
    #data_dir = "./recaptcha-dataset/Large"
    class_names = ['Bicycle', 'Bridge', 'Bus', 'Car', 
                'Chimney', 'Crosswalk', 'Hydrant', 
                'Motorcycle', 'Palm', 'Traffic Light']

    data_transforms = transforms.Compose([
            transforms.ToTensor(),                      
            transforms.RandomResizedCrop(input_size),   
            transforms.RandomHorizontalFlip(),         
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])

    print("Initializing Datasets and Dataloaders...")

    image_datasets = datasets.ImageFolder(dataset_dir, data_transforms)  
    num_data = len(image_datasets)
    indices = np.arange(num_data)
    np.random.shuffle(indices)  # index만 셔플

    train_size = int(num_data*0.8)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_set = torch.utils.data.Subset(image_datasets, train_indices)
    val_set = torch.utils.data.Subset(image_datasets, val_indices)

    print('Number of training data:', len(train_set))
    print('Number of validation data:', len(val_set))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, val_loader
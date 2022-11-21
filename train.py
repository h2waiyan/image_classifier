import argparse
import torch
from torch import nn, optim, cuda
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
import os

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',type=str, help='Path of directory with data to train and test')
    parser.add_argument('--arch',action='store',type=str, default = 'densenet121', help='Pretrained networks for the model. e.g., vgg16, and densenet121')
    parser.add_argument('--hidden_units',action='store',type=int, default = '512', help='Hidden units for the classifier')
    parser.add_argument('--learning_rate',action='store',type=float, default = '0.003', help='Learning rate for the model')
    parser.add_argument('--epochs',action='store',type=int, default = '1', help='Number of epochs you want to perform gradient descent')
    parser.add_argument('--save_dir', type=str, help='Name of file to save the trained model')
    parser.add_argument('--gpu',action='store_true',help='Use GPU if available')

    args = parser.parse_args()
    
    device = "cpu"

    if args.arch:
        arch = args.arch
        if arch is not "vgg16" and arch is not "densenet121":
            raise Exception('Oops! This architecture is not supported');
        
    if args.hidden_units:
        hidden_units = args.hidden_units
        if arch == "vgg16":
            if hidden_units >= 25088:
                raise Exception('Oops! Hidden units for VGG16 Architecture must be 256 < hidden_units < 25088')
            elif hidden_units <= 256:
                raise Exception('Oops! Hidden units for VGG16 Architecture must be 256 < hidden_units < 25088')
                
        else:
            if hidden_units >= 1024:
                raise Exception('Oops! Hidden units for VGG16 Architecture must be 256 < hidden_units < 1024')
            elif hidden_units <= 256:
                raise Exception('Oops! Hidden units for VGG16 Architecture must be 256 < hidden_units < 1024')
        
        
    if args.learning_rate:
        learning_rate = args.learning_rate
        
    print(args.epochs)    
    if args.epochs:
        epochs = args.epochs
        print(epochs)
        if epochs < 1:
            raise Exception('Oops! Epochs must be > 1.')
    else:
        raise Exception('Oops! Epochs must be > 1.')        
            
    if args.gpu:        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = 'checkpoint.pth'
        
    print(device)
     
    
    data_dir = args.data_dir
    
    if not os.path.exists(data_dir):
        raise Exception('Oops! Your training folder does not exist!');
        
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    class_to_idx = train_data.class_to_idx

    criterion = nn.NLLLoss()
    
    model, optimizer = load_model(arch, hidden_units, learning_rate)
    
    loaded_model = model.to(device)
    
    print('------')
    print('Loading Complete')
    print('------')
    
    trained_model = train_model(epochs, loaded_model, trainloader, validloader, device, criterion, optimizer)

    print('------')
    print('Training Complete')
    print('------')
    
    save_model(trained_model, arch, save_dir, optimizer, class_to_idx)

    print('------')
    print('Model Saved Successfully.')
    print('------')

def load_model(arch, hidden_units,learning_rate):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088,hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units,256)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(256, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
        
    else:
        model = models.densenet121(pretrained=True)
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024,hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units,256)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(256, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = classifier
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    
    return model, optimizer

def train_model(epochs, model, trainloader, validloader, device, criterion, optimizer):
    
    steps = 0
    running_loss = 0
    print_every = 10
    
    print('------')
    print('Training Starts.')
    print('------')
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                valid_accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train loss: {running_loss/print_every:.3f} | "
                      f"Valid loss: {valid_loss/len(validloader):.3f} | "
                      f"Valid accuracy: {valid_accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
            
    return model

def save_model(model, arch, save_dir, optimizer, class_to_idx):
        
    checkpoint = {
    'arch' : arch,
    'state_dict': model.state_dict(),
    'optimizer_dict': optimizer.state_dict(),
    'class_to_idx': class_to_idx,
    'classifier': model.classifier,
        }
    
    torch.save(checkpoint, save_dir)
    
if __name__ == "__main__":
    main()
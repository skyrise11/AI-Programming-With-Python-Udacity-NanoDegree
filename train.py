



import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
from collections import OrderedDict
import os
import copy

parser = argparse.ArgumentParser(description='Train Image Classifier')

parser.add_argument('data_directory')
parser.add_argument('--save_dir', action='store', default='.')
parser.add_argument('--arch', action='store', default='vgg19')
parser.add_argument('--learning_rate', action='store', default=0.01, type=float)
parser.add_argument('--hidden_units', default=512, type=int)
parser.add_argument('--epochs', action='store', default=10, type=int)
parser.add_argument('--gpu', action='store_true', default=False)

args = parser.parse_args()


data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

incorrect_data_dir = not os.path.exists(train_dir) \
                     or not os.path.exists(valid_dir) \
                     or not os.path.exists(test_dir)

if incorrect_data_dir:
    print('**Alert! There is no such directory existing Alert!** ' + args.data_directory + '/n')
    exit(1)

data_transforms = {
    'train': transforms.Compose([transforms.Resize(256), transforms.RandomRotation(30), transforms.CenterCrop(224),
                                 transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                  ['train', 'valid', 'test']}

batch_size = 64
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for
               x in ['train', 'valid', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
class_names = image_datasets['train'].classes

init_lr = args.learning_rate
epochs = args.epochs
model_nm = args.arch


if model_nm == 'densenet161':
    model = models.densenet161(pretrained=True)
    input_features = 2208
    # print(model)
elif model_nm == 'vgg19':
    model = models.vgg19(pretrained=True)
    input_features = 25088
else:
    print(model_nm + "Not trained model: Can only choose 'vgg19 or 'densenet161'")

for param in model.parameters():
    param.requires_grad = False

from torch.optim import lr_scheduler



classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_features, args.hidden_units)),
    ('relu', nn.ReLU()),
    # ('droput',nn.Dropout(0.5)),
    ('fc2', nn.Linear(args.hidden_units, 102)),
    ('output', nn.Softmax(dim=1))
]))

model.classifier = classifier

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=init_lr)
sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

processor_type = 'cuda:0' if args.gpu and torch.cuda.is_available() else 'cpu'
device = torch.device(processor_type)
model.to(device)


# Below code is for the cuda "images.cuda()"

# Referrences used from the link "https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html" and the link was suggested from previous occurences and in the communities.

def train_model(model, criterion, optimizer, sched, num_epochs=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 15)

        # Below is for the each entitty of the epoch, will have both training and validation phases altogether \.

        for phase in ['train', 'valid', 'test']:
            print(phase)
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            # Iteration is happening over data
            for inputs, labels in dataloaders[phase]:
                print(labels)
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # Best way to illustrate the going forward and tracking the history, while in train

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Also same illustration done with the backward and with optimizing during the training period.


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Below will be the logic to find the statistics of Accuracy and the correctness.

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # below is for the complete copy of the Accuracy

            if phase == 'valid':
                print('{} Accuracy:{:.4f}'.format(phase, epoch_accuracy))
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Accuracy: {:4f}'.format(best_acc))

    # Below is to Load the weights of the best model

    model.load_state_dict(best_model_wts)

    return model


print('No. of Epochs: ', args.epochs)
print('Learning rate: ', args.learning_rate)
print('Pre-trained model: ', args.arch)
print('Classification layers: ', args.hidden_units)
print('Start of the Image Classification model')


model.to(device)
model = train_model(model, criterion, optimizer, sched, epochs)
print('Epochs started')


# TODO: Do validation on the test set

model.eval()

accuracy = 0

for inputs, labels in dataloaders['test']:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)

    # The below used class is with the best possibility which predicts the predicted class.

    equality = (labels.data == outputs.max(1)[1])

    # Accuracy of the model is "number of predictions" (That are correct) / "All predictions."

    accuracy += equality.type_as(torch.FloatTensor()).mean()

print("Accuracy of the model on Test Data: {:.3f}".format(accuracy / len(dataloaders['test'])))

model.class_to_idx = image_datasets['train'].class_to_idx


def init_checkpoint():
    checkpoint = {
        'hidden_units': args.hidden_units,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'batch_size': 32,
        'output_size': 102,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'arch': args.arch,
        'scheduler': sched,
        'checkpoint.pth': args.save_dir,
        'classifier': classifier,
    }

    torch.save(checkpoint, checkpoint.pth)


init_checkpoint()
my_saved_models = 'myclassifier.pth'
path = "/content/gdrive/My Drive/Colab_Notebooks/my_saved_models"
torch.save(model.state_dict(), path)
checkpoint = torch.load('checkpoint.pth')
checkpoint.keys()


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']

    for param in model.parameters():
        param.requires_grad = False

    return model, checkpoint['class_to_idx']


model, class_to_idx = load_checkpoint('./checkpoint.pth')
model
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd    
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm

def submission(taskId, answer):
    with open("submission.csv", 'a+') as f:
        f.seek(0)
        if not any(line.strip() == "taskId,answer" for line in f):
            f.write("taskId,answer" + "\n")
        f.write(taskId + "," + str(int(answer)) + "\n")

# Проверить доступность CUDA и выбрать устройство
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Код для тестирования модели с использованием F1-меры
def test_model(model, test_dataloader):
    model.eval()  # Перевести модель в режим оценки
    true_labels = []
    predicted_labels = []
    with torch.no_grad():  # Отключить вычисление градиентов
        for img0, img1, label in test_dataloader:
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            output1, output2 = model(img0, img1)
            # Вычислить расстояние между векторами и преобразовать его в предсказанные метки
            distances = F.pairwise_distance(output1, output2)
            predicted = (distances < 0.5).float()  # Пороговое значение для классификации
            true_labels.extend(label.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
    f1 = f1_score(true_labels, predicted_labels)
    return f1

# Showing images
def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

# Plotting data
def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

def modelPath(epoch = 1):
    return f"models/siamese_model_epoch_{epoch}.pth"
    
# preprocessing and loading the dataset
class SiameseDataset:
    def __init__(self, training_csv=None, training_dir=None, transform=None):
        # used to prepare the labels and images path
        self.train_df = pd.read_csv(training_csv)
        self.train_df.columns = ["taskId", "label"]
        self.train_dir = training_dir
        self.transform = transform

    def __getitem__(self, index):

        # getting the image path
        image1_path = f"{os.path.join(self.train_dir, self.train_df.iat[index, 0])}/image1.jpg"
        image2_path = f"{os.path.join(self.train_dir, self.train_df.iat[index, 0])}/image2.jpg"
        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return (
            img0,
            img1,
            torch.from_numpy(
                np.array([1-int(self.train_df.iat[index, 1])], dtype=np.float32)
            ),
            self.train_df.iat[index, 0]
        )

    def __len__(self):
        return len(self.train_df)
    
#create the Siamese Neural Network

class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,2)
        )
        
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2
    
# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidean distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive
    
if __name__ == '__main__':

    # Load the training dataset
    folder_dataset = datasets.ImageFolder(root="dataset2/train/")

    # Resize the images and transform to tensors
    transformation = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor()])

    # Initialize the network
    #siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset, transform=transformation)
    
    siameseTrainDataset = SiameseDataset(training_dir = "datasets/train", training_csv = "datasets/train/submission.csv", transform=transformation)
    siameseTestDataset = SiameseDataset(training_dir = "datasets/val", training_csv = "datasets/val/submission.csv", transform=transformation)

    # Create a simple dataloader just for simple visualization
    vis_dataloader = DataLoader(siameseTrainDataset,
                            shuffle=True,
                            num_workers=2,
                            batch_size=8)

    # Extract one batch
    example_batch = next(iter(vis_dataloader))

    # Example batch is a list containing 2x8 images, indexes 0 and 1, an also the label
    # If the label is 1, it means that it is not the same person, label is 0, same person in both images
    concatenated = torch.cat((example_batch[0], example_batch[1]),0)

    #print(example_batch[2].numpy().reshape(-1))
    #imshow(torchvision.utils.make_grid(concatenated))
    print(f"CUDA: {torch.cuda.is_available()}")



    # Load the training dataset
    trainDataloader = DataLoader(siameseTrainDataset,
        shuffle=True,
        batch_size=64,
        num_workers=8,
        pin_memory=True)
    
    testDataloader = DataLoader(siameseTestDataset,
        shuffle=True,
        batch_size=64,
        num_workers=8,
        pin_memory=True)

    net = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.005 )
    num_epochs = 50

    counter = []
    loss_history = [] 
    iteration_number= 0
    graphArr = []
    # Initialize model

    testDataloader = DataLoader(siameseTestDataset, num_workers=2, batch_size=1, shuffle=False)
    net.load_state_dict(torch.load(modelPath(num_epochs)))

    # Grab one image that we are going to test
    ans = 0

    for i, (img0, img1, label ,taskId) in tqdm(enumerate(testDataloader, 0)):
        # Iterate over 5 images and test them with the first image (x0)
        x0, x1, label2, taskId = img0, img1, label, taskId
        # Concatenate the two images together
        concatenated = torch.cat((x0, x1), 0)
        
        output1, output2 = net(x0.cuda(), x1.cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        if(euclidean_distance.item() < 0.8):
            ans = 0
        else:
            ans = 1
        
        ansId = str(taskId)[2:-3]

        try:
            submission(ansId, ans)
        except:
            pass

        #imshow(torchvision.utils.make_grid(concatenated), f'Dissmlty: {euclidean_distance.item():.2f} Label: {label2} Ans: {ans}')



        #imshow(torchvision.utils.make_grid(concatenated), f'Label: {label2} Answer: {ans}')


    '''
    for epoch in range(1, 100):
        net.load_state_dict(torch.load(modelPath(epoch)))

        # Test the model
        graphArr.append(test_model(net, trainDataloader))
        print(f"F1-score of Epoch {epoch}: {graphArr[-1]}")
                    

    show_plot(range(1, 100), graphArr)
    '''



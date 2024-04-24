from tqdm import tqdm
import json
from time import sleep
import requests
from pathlib import Path

def getImage(url):
    try:
        response = requests.get(url) 
        img = response.content
        return img
    except:
        sleep(5)

def createFolders():
    try:
        Path("dataset2/test/True").mkdir(parents=True)
    except FileExistsError:
        pass
    try:
        Path("dataset2/val/True").mkdir(parents=True)
    except FileExistsError:
        pass
    try:
        Path("dataset2/train/True").mkdir(parents=True)
    except FileExistsError:
        pass
    try:
        Path("dataset2/test/False").mkdir(parents=True)
    except FileExistsError:
        pass
    try:
        Path("dataset2/val/False").mkdir(parents=True)
    except FileExistsError:
        pass
    try:
        Path("dataset2/train/False").mkdir(parents=True)
    except FileExistsError:
        pass

def saveData(typeOfData):
    if((typeOfData != val_data) and (typeOfData != test_data) and (typeOfData != train_data)):
        raise ValueError("Incorrect typeOfData")
    
    for result in tqdm(typeOfData["data"]["results"]):
        taskId = result["taskId"]
        if(typeOfData != val_data):
            isItReallyDuplicates = bool(int(result["answers"][0]["answer"][0]["id"]))
        image1_url = result["representativeData"]["image1"]["imageUrl"]
        image2_url = result["representativeData"]["image2"]["imageUrl"]
        
        image1 = getImage(image1_url)
        image2 = getImage(image2_url)

        if((typeOfData == test_data) and isItReallyDuplicates == True):
            path = f"dataset2/test/True/{taskId}"
            try:
                Path(path).mkdir(parents=True)
            except FileExistsError:
                pass
        elif((typeOfData == test_data) and isItReallyDuplicates == False):
            path = f"dataset2/test/False/{taskId}"
            try:
                Path(path).mkdir(parents=True)
            except FileExistsError:
                pass
        
        elif((typeOfData == train_data) and isItReallyDuplicates == True):
            path = f"dataset2/train/True/{taskId}"
            try:
                Path(path).mkdir(parents=True)
            except FileExistsError:
                pass
        elif((typeOfData == train_data) and isItReallyDuplicates == False):
            path = f"dataset2/train/False/{taskId}"
            try:
                Path(path).mkdir(parents=True)
            except FileExistsError:
                pass
        
        elif(typeOfData == val_data):
            path = f"dataset2/val/{taskId}"
            try:
                Path(path).mkdir(parents=True)
            except FileExistsError:
                pass

        try:
            with open(f"{path}/image1.jpg", 'wb') as f:
                f.write(image1)

            with open(f"{path}/image2.jpg", 'wb') as f:
                f.write(image2)
        except:
            sleep(10)
            image1 = getImage(image1_url)
            image2 = getImage(image2_url)
            
            with open(f"{path}/image1.jpg", 'wb') as f:
                f.write(image1)

            with open(f"{path}/image2.jpg", 'wb') as f:
                f.write(image2)

#path to train/test data
test_data = json.load(open("jsons/test_task1.json"))
train_data = json.load(open("jsons/train_task1.json"))
val_data = json.load(open("jsons/val_task1.json"))

createFolders()

saveData(test_data)
saveData(train_data)
saveData(val_data)

print("\n\n Done")


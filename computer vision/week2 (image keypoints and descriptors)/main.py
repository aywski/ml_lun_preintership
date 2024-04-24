import json
import cv2
import numpy as np
import requests
from time import sleep 
from sklearn.metrics import f1_score
import os


#path to train/test data
with open('jsons/test_task1.json', 'r') as test_task1:
    testData = json.load(test_task1)
    
with open('jsons/train_task1.json', 'r') as train_task1:
    trainData = json.load(train_task1)

with open('jsons/val_task1.json', 'r') as val_task1:
    valData = json.load(val_task1)

#renew log/sub files
with open('data.log', 'w') as file:
    file.write('')

with open('mysubmission.csv', 'w') as file:
    file.write('')    

def log(line_to_add = "", file_path = "data.log"):
    with open(file_path, 'a') as f:
        f.write(line_to_add + "\n")

def submission(taskId, answer):
    with open("mysubmission.csv", 'a+') as f:
        f.seek(0)
        if not any(line.strip() == "taskId,answer" for line in f):
            f.write("taskId,answer" + "\n")
        f.write(taskId + "," + str(int(answer)) + "\n")

def getImage(url):
    try:
        response = requests.get(url) 
        nparr = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except:
        sleep(5)

def resize_image(img, maxSize):
    while img.shape[1] > maxSize or img.shape[0] > maxSize:
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    return img

def SIFTAlgorithm(img1, img2, maxSize = 2000):
    if((img1.shape[1]) or (img1.shape[0]) or (img2.shape[1]) or (img2.shape[0])) > maxSize:
        img1 = resize_image(img1, maxSize)
        img2 = resize_image(img2, maxSize)
    image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(enable_precise_upscale = True)
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2,None)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(descriptors_1,descriptors_2 )
    matches = sorted(matches, key = lambda x:x.distance)
    return len(matches), len(keypoints_1), len(keypoints_2)

def algMoreAccuracyData(sift):
    if(sift > 0.2):
        return True
    else:
        return False

totalCountOfData = 0
countOfDuplicates = 0
countOfCorrectAns = 0

avOfSift = 0
avOfOrb = 0
avOfAmount = 0

arrayReallyAns = []
arrayReallyOfMyAns = []

#Parse images from JSON
for result in valData["data"]["results"]:
    
    totalCountOfData = totalCountOfData + 1
    
    taskId = result["taskId"]
#    isItReallyDuplicates = bool(int(result["answers"][0]["answer"][0]["id"]))
#    arrayReallyAns.append(int(isItReallyDuplicates))
    image1_url = result["representativeData"]["image1"]["imageUrl"]
    image2_url = result["representativeData"]["image2"]["imageUrl"]
    
    image1 = getImage(image1_url)
    image2 = getImage(image2_url)
        
    os.system('cls')
    
    retry = 10
    while retry:
        try:
            siftAns = SIFTAlgorithm(image1, image2)
            retry = 0
        except:
            retry -= 1

    tempnum = 0
    if(siftAns[1] < siftAns[2]):
        tempnum = siftAns[1]
    else:
        tempnum = siftAns[2]

    siftAccuracy = siftAns[0] / tempnum
        
    answer = algMoreAccuracyData(siftAccuracy)
    arrayReallyOfMyAns.append(int(answer))
    accuracy = int(countOfCorrectAns * 100 / totalCountOfData)

    print("-" * 100)

    print(f"SIFT Matches / Keypoint 1 / Keypoint 2: {siftAns[0]} / {siftAns[1]} / {siftAns[2]}")
    print(f"Accuracy of SIFT: {siftAccuracy}")

    print(f"TaskId: {taskId}")
    print(f"Image1 URL: {image1_url}")
    print(f"Image2 URL: {image2_url}")
    print(f"Total duplicate found: {countOfDuplicates} / {totalCountOfData}")
    print(f"Correct answers: {countOfCorrectAns} / {totalCountOfData}")

    print(f"My Answer: {answer}")
#    print(f"Real Answer: {isItReallyDuplicates}")
    print(f"My Accuracy: {accuracy}%")
#    print(f"F1 Score: {f1_score(arrayReallyAns, arrayReallyOfMyAns, average='micro')}")

    print("-" * 100)
#    log(f"isItReallyDub?:{isItReallyDuplicates}  siftAcc:{siftAccuracy}  id:{taskId}")
    submission(taskId, int(answer))

print("\n\n :)")
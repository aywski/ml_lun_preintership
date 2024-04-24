import json
import cv2
from PIL import Image
import imagehash
import requests
import datetime
import time

#path to train/test data
with open('test_task1.json', 'r') as test_task1:
    testData = json.load(test_task1)
    
with open('train_task1.json', 'r') as train_task1:
    trainData = json.load(train_task1)

with open('val_task1.json', 'r') as val_task1:
    valData = json.load(val_task1)

#renew log/sub files
with open('data.log', 'w') as file:
    file.write('')

with open('mysubmission.csv', 'w') as file:
    file.write('')    

#hsv method
def calculate_hsv_similarity(image1, image2):
    # Преобразование изображений в формат HSV
    hsv_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hsv_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

    # Рассчет среднего значения по каждому каналу HSV
    mean_hsv1 = cv2.mean(hsv_image1)[:3]
    mean_hsv2 = cv2.mean(hsv_image2)[:3]

    # Рассчет евклидова расстояния между средними значениями
    hsv_similarity = cv2.norm(mean_hsv1, mean_hsv2, cv2.NORM_L2)

    return 1 - hsv_similarity/100

#phash method
def calculate_phash_similarity(image1, image2):
    # Преобразование изображений в grayscale
    grayscale_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayscale_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Вычисление хешей для изображений
    hash1 = imagehash.phash(Image.fromarray(grayscale_image1))
    hash2 = imagehash.phash(Image.fromarray(grayscale_image2))

    # Приведение хешей к одинаковой длине
    hash1.hash = hash1.hash[:len(hash2.hash)]
    hash2.hash = hash2.hash[:len(hash1.hash)]

    # Рассчет хеш-различия между изображениями
    phash_difference = hash1 - hash2
    phash_similarity = 1.0 - phash_difference / max(len(hash1.hash) * 8, 1)

    return phash_similarity

#total calculations 
def calculate_combined_similarity(image1, image2, hsv_weight=0.7, phash_weight=0.3):
    # Рассчет сходства по HSV
    hsv_similarity = calculate_hsv_similarity(image1, image2)

    # Рассчет сходства по pHash
    phash_similarity = calculate_phash_similarity(image1, image2)

    # Комбинирование результатов с учетом весов
    combined_similarity = hsv_weight * hsv_similarity + phash_weight * phash_similarity

    return combined_similarity

def log(line_to_add = "", file_path = "data.log"):
    with open(file_path, 'a') as f:
        f.write(line_to_add + "\n")

def submission(taskId, answer):
    with open("mysubmission.csv", 'a+') as f:
        f.seek(0)
        if not any(line.strip() == "taskId,answer" for line in f):
            f.write("taskId,answer" + "\n")
        f.write(taskId + "," + str(int(answer)) + "\n")

def algMoreAccuracyData(hsv, phash): #suspicion of similarity
    if(phash >= 0.6 or hsv >= 0.65):
        return True
    else:
        return False
        

totalCountOfData = 0
countOfDuplicates = 0
countOfCorrectAns = 0
currentTime = datetime.datetime.now()

#Parse images from JSON
for result in valData["data"]["results"]:
    totalCountOfData = totalCountOfData + 1
    
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'}
    taskId = result["taskId"]
    image1_url = result["representativeData"]["image1"]["imageUrl"]
    image2_url = result["representativeData"]["image2"]["imageUrl"]
    
    try: 
        img1_data = requests.get(image1_url, headers=headers).content
        img2_data = requests.get(image2_url, headers=headers).content
    except:
        time.sleep(5)
        continue

    with open('image1.jpg', 'wb') as f:
        f.write(img1_data)
    
    with open('image2.jpg', 'wb') as f:
        f.write(img2_data)


    image1 = cv2.imread("image1.jpg")
    image2 = cv2.imread("image2.jpg")
    
    hsv = calculate_hsv_similarity(image1, image2)
    phash = calculate_phash_similarity(image1, image2)
    averageSimilarity = calculate_combined_similarity(image1, image2, hsv_weight=0.7, phash_weight=0.3)
    accuracy = int(countOfCorrectAns * 100 / totalCountOfData)
    answer = algMoreAccuracyData(hsv, phash)

    print(f"TaskId: {taskId}")
    print(f"Image1 URL: {image1_url}")
    print(f"Image2 URL: {image2_url}")
    print(f"Total duplicate found: {countOfDuplicates} / {totalCountOfData}")
    print(f"Correct answers: {countOfCorrectAns} / {totalCountOfData}")

    print(f"Time (Start/Now): {currentTime.hour}:{currentTime.minute} / {datetime.datetime.now().hour}:{datetime.datetime.now().minute}")

    print(f"Score of probality dublicate of hsv: {hsv}")
    print(f"Score of probality dublicate of phash: {phash}")
    print(f"Score of probality dublicate average: {averageSimilarity}")

    print(f"My Answer: {answer}")
    print(f"My Accuracy: {accuracy}%")
    
    print("-" * 100)
    log(f"myResult:{averageSimilarity}\tmyAccuracy:{accuracy}%\thsv:{hsv}\tphash{phash}\tid:{taskId}")
    submission(taskId, answer)

print("\n\n :)")
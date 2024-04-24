import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef

def clear_csv_file(filename):
  """
  Clears the contents of a CSV file.

  Args:
      filename (str): The name of the CSV file.

  Raises:
      OSError: If an error occurs while opening or writing to the file.
  """

  try:
    with open(filename, 'w', encoding='utf-8') as f:
      # Write an empty string to effectively clear the file
      f.write("")
  except OSError as e:
    raise OSError(f"An error occurred while clearing the file '{filename}': {e}") from e

# Example usage
clear_csv_file("train.csv")

def csvFileWithData(taskId, geo_region, geo_city, geo_district, central_city, geo_street, geo_address, geo_building, geo_region_oblast, geo_microdistrict):
    with open("train.csv", 'a+', encoding='utf-8') as f:
        f.seek(0)
        if not any(line.strip() == "taskId,geo_region,geo_city,geo_district,central_city,geo_street,geo_address,geo_building,geo_region_oblast,geo_microdistrict" for line in f):
            f.write("taskId,geo_region,geo_city,geo_district,central_city,geo_street,geo_address,geo_building,geo_region_oblast,geo_microdistrict" + "\n")
        f.write(taskId + "," + geo_region + "," + geo_city + "," + geo_district + "," + central_city + "," + geo_street + "," + geo_address + "," + geo_building + "," + geo_region_oblast + "," + geo_microdistrict + "\n")

#path to jsons
with open('jsons/test_geo_extractor.json', 'r', encoding='utf-8') as test_geo:
    testData = json.load(test_geo)
    
with open('jsons/train_geo_extractor.json', 'r', encoding='utf-8') as train_geo:
    trainData = json.load(train_geo)

with open('jsons/val_no_answer_geo_extractor.json', 'r', encoding='utf-8') as val_geo:
    valData = json.load(val_geo)

#data for use
data = trainData

for i, result in enumerate(data["data"]["results"]):
    #print(result["answers"][0]["answer"])
    taskId = result["taskId"]
    
    print("\n", "-"*30)
    print(f"Task id: {taskId}")
    geo_region = geo_city = geo_district = central_city = geo_street = geo_address = geo_building = geo_region_oblast = geo_microdistrict = " "
    for id in result["answers"][0]["answer"]:
        geoTag = id["id"]
        print(f"\n{geoTag}:")
        
        geoAns = ""
        for i in range (id["data"][0]["start"], id["data"][0]["end"]):
            geoAns += result["representativeData"]["page_data_words"][i] + " "
            #print(result["representativeData"]["page_data_words"][i])
        
        geoAns = geoAns[:-1].replace(",", "")
        print(geoAns)
        
        geo_region = geoAns if geoTag == "geo_region" else geo_region
        geo_city = geoAns if geoTag == "geo_city" else geo_city 
        geo_district = geoAns if geoTag == "geo_district" else geo_district 
        central_city = geoAns if geoTag == "central_city" else central_city 
        geo_street = geoAns if geoTag == "geo_street" else geo_street 
        geo_address = geoAns if geoTag == "geo_address" else geo_address 
        geo_building = geoAns if geoTag == "geo_building" else geo_building 
        geo_region_oblast = geoAns if geoTag == "geo_region_oblast" else geo_region_oblast 
        geo_microdistrict = geoAns if geoTag == "geo_microdistrict" else geo_microdistrict 
    csvFileWithData(taskId, geo_region, geo_city, geo_district, central_city, geo_street, geo_address, geo_building, geo_region_oblast, geo_microdistrict)

import os
import pandas as pd

directory = os.getcwd()
extensions = ('jpeg', 'jpg', 'png', 'tif')
ext = []
subfolders = [folder.path for folder in os.scandir() if os.path.isdir(folder)]
df = pd.DataFrame(columns = ['v1', 'v2'])
count = 0

  
# function to get unique values
def unique(list1):
  
    # initialize a null list
    unique_list = []
      
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    for x in unique_list:
        print(x),
      

for subdir in subfolders:
    for folder in os.walk(subdir):
        for item in folder[2]:
            # if item.lower().endswith(extensions)
            split_string = item.split('.')
            
            if split_string[-1].lower() in extensions:
                count += 1

            # Get all extensions to check what image ext are used
            #ext.append(split_string[-1].lower())

print("Number of image files: ", count)            



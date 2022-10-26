import os
from zipfile import ZipFile
import shutil

year = input("What year are you sorting?: ")
keywords = [year, 'wetransfer']
subfolders = []
zip_files = './All zip files'

def extract_folders(year):
    
    print('Selecting correct folders...')
    for f in os.scandir():
        count = 0
        for word in keywords:
                if word in f.path:
                    count += 1
        if count == len(keywords):
            subfolders.append(f.path)

    print("Extracting files from selected folders...")
    for path in subfolders:  
        # open the zip file in read mode
        with ZipFile(path, 'r') as zip: 
            # extract all files to another directory
            zip.extractall(f"../sortedData/{year}")
            print(f"Extracted files from: {path}")
        
        print("Moving zip folder...")
        shutil.move(path, zip_files)


extract_folders(year)


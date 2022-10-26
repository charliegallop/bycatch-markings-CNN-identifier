# https://towardsdatascience.com/convert-pascal-voc-xml-to-yolo-for-object-detection-f969811ccba5

import random
import glob
import os
import shutil



class SplitDatasets():

    def __init__(self, images_dir, labels_dir, resize_to, set_dirs, backbone, train_for):
        self.lower_limit = 0
        self.backbone = backbone
        self.train_for = train_for

        # Directory to copy images from
        self.images_dir = images_dir

        # Directory to copy labels from
        self.labels_dir = labels_dir

        self.resize_to = resize_to
        
        self.image_paths = self.get_image_paths()

        self.set_splits = {"train": 0.7, "test": 0, "valid": 0.30}
        self.set_paths = {"train": set_dirs['train'], "test": set_dirs['test'], "valid": set_dirs['valid']}

    def get_image_paths(self):
        return glob.glob(os.path.join(self.images_dir, '*.jpg'))
    
    def get_label_paths(self, label_dir):
        return glob.glob(os.path.join(label_dir, "*.xml"))

    def prompt_split(self):
        temp = input("Do you want to reshuffle the dataset? [y][n] ").lower()
        if temp == 'y':
            return True
        else:
            return False

    def remove_small_images(self):
        from PIL import Image
        for image in self.image_paths:
            img=Image.open(image)
            if (img.size[0] < self.resize_to) and (img.size[1] < self.resize_to):
                self.image_paths.remove(image)
                print("REMOVING: ", os.path.basename(image), "since size is: ", img.size)

    def shuffle_paths(self):
        # Shuffle paths
        return random.Random(4).shuffle(self.image_paths)

    def check_sum_split(self):
        check_sum = sum([self.set_splits[x] for x in self.set_splits])
        assert check_sum == 1.0, "Split proportion is not equal to 1.0"

    def copyfiles(self, fil, set_dir):
        global master_dir
        basename = os.path.basename(fil)
        filename = os.path.splitext(basename)[0]
        
        # image
        src = fil
        dest = os.path.join(set_dir, "images", f"{filename}.jpg")
        shutil.copyfile(src, dest)  

        # label
        src = os.path.join(self.labels_dir, f"{filename}.xml")
        dest = os.path.join(set_dir, "labels", f"{filename}.xml")
        if os.path.exists(src):
            shutil.copyfile(src, dest)


    def check_dir_already_exists(self, set_path = None):
        safe_to_create_dir = False
        existing_dir = []

        if os.path.exists(set_path):
            user_input = input(f"Remove '{set_path}' directory and contents? [y][n]: ").lower()
            if user_input == "y":
                shutil.rmtree(set_path)
                print("Removed directory!")
                safe_to_create_dir = True
            else:
                print("Did not delete directory...")
                print("Not safe to create new directories...Quitting...")
                quit()
        else:
            print(f"'{set_path}' directory does not already exist...")
            safe_to_create_dir = True

        if safe_to_create_dir:
            print("Safe to create new directories")
            return safe_to_create_dir
        else:
            print("Not safe to create new directories")
            return safe_to_create_dir


    def create_set_directories(self, set_path):
        os.mkdir(set_path)
        set_label_dir = os.path.join(set_path, "labels")
        set_image_dir = os.path.join(set_path, "images")
        os.mkdir(set_label_dir)
        os.mkdir(set_image_dir)

    def split_files(self, lower_limit, set_splits):
        for se in set_splits:
            limit = round(len(self.image_paths) * set_splits[se])
            set_path = self.set_paths[se]
            for image in self.image_paths[lower_limit:lower_limit + limit]:
                self.copyfiles(fil = image, set_dir = set_path)
            lower_limit = lower_limit + limit


    def main(self):
        if self.prompt_split():
            for sp in self.set_paths:
                proceed = self.check_dir_already_exists(self.set_paths[sp])
            if proceed:
                print("SAFE TO PROCEED")

                for se in self.set_splits:
                    print("EXECUTING CREATE")
                    self.create_set_directories(self.set_paths[se])

                print("SPLIT CREATE")
                self.remove_small_images()
                self.split_files(self.lower_limit, self.set_splits)

                # Creates XML files with edited classes i.e. removes dolphin if training for markings
                # This is so it can be used to work out the metrics
                # Copies from validation directory
                eval_dir = "/home/charlie/Documents/Uni/Exeter - Data Science/Dissertation/data/processed/evaluation"
                WRITE_XML_TO = os.path.join(eval_dir, self.train_for, self.backbone, "gt")
                from edit_xml import EditXml
                save_edited_xmls = EditXml(
                    write_to = WRITE_XML_TO,
                    train_for = self.train_for,
                    labels_dir = self.set_paths['valid'])
                save_edited_xmls.run()

            else:
                print("Directories already exist")
        else:
            print("Not shuffling data, keeping existing folders")
    

from xml.etree import ElementTree as et
import glob
import os

class EditXml():
    def __init__(self, write_to, labels_dir, train_for):
        self.write_to = write_to
        self.labels_dir = labels_dir
        self.train_for = train_for
        self.label_paths = glob.glob(f"{self.labels_dir}/labels/*")

    def run(self):
        if self.train_for == 'dolphin':
            self.keep_dolphin()
        elif self.train_for == 'markings':
            self.keep_markings()
        elif self.train_for == 'all':
            self.keep_all()


    def keep_dolphin(self):
        print("Making XML files for 'dolphin'....")
        for i in self.label_paths:
            image_name = os.path.basename(i)
            tree = et.parse(i)
            root = tree.getroot()
            for member in root.findall('object'):
                if (member.find('name').text != 'dolphin'):
                    if (member.find('name').text != 'background'):
                        root.remove(member)
            save_as = os.path.join(self.write_to, image_name)
            tree.write(save_as)
        print("XML files saved to: ", self.write_to)
    
    def keep_markings(self):
        print("Making XML files for 'markings'....")
        for i in self.label_paths:
            image_name = os.path.basename(i)
            tree = et.parse(i)
            root = tree.getroot()

            for member in root.findall('object'):
                if member.find('name').text == 'dolphin':
                    root.remove(member)

            save_as = os.path.join(self.write_to, image_name)
            tree.write(save_as)
        print("XML files saved to: ", self.write_to)
    
    def keep_all(self):
        print("Making XML files for 'markings'....")
        for i in self.label_paths:
            image_name = os.path.basename(i)
            tree = et.parse(i)
            root = tree.getroot()
            save_as = os.path.join(self.write_to, image_name)
            tree.write(save_as)
        print("XML files saved to: ", self.write_to)

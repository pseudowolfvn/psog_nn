import os

class ImgPathGenerator:
    def __init__(self, subj_root):
        self.root = os.path.join(subj_root, 'images')
        self.img_ind = 0

    def __next__(self):
        while os.path.exists(self.get_img_path()):
            img_path = self.get_img_path()
            self.img_ind += 1
            return img_path
        raise StopIteration()
    
    def __iter__(self):
        return self

    def get_img_path(self):
        return os.path.join(self.root, str(self.img_ind) + '.jpg')

    def get_root(self):
        return self.root
import numpy as np
import random
import copy
from PIL import Image
from ast import literal_eval
from scipy import misc, ndimage


class Dataset:
    """
    NOTE: Does not check if a image returned exist.
    """

    def __init__(self, path, num_images, batch_size, class_id_files_path, multitask=False, data_augmentation=True,
                 ignore_first_row=False, shuffle=True):
        """
        Args:
            num_images: Integer
            validation: If true, no data augmanetation is performed

        Attributes:
            epoch: If you are in epoch 1, it means that you still have samples to request for, for the first time.
        
        Note: multitask option is not compatible with bounding boxes yet
        """
        self.classesID = self.parse_classID_file(class_id_files_path)
        self.images = self.parse_file(path, ignore_first_row)
        self.num_images = len(self.images)
        self.batch_size = batch_size
        self.current_epoch_images = []
        self.epoch = -1
        self.shuffle = shuffle
        self.new_epoch()
        self.data_augmentation = data_augmentation
        self.multitask = multitask
        assert self.num_images == num_images, "Number of images in %s is %i, different of argument num_images= %i" % (
        path, self.num_images, num_images)

    def parse_classID_file(self, path):
        """
        ARGS:
        """
        labels = np.array(np.loadtxt(path, str, delimiter='\t'))
        labelsDICT = dict([(int(literal_eval(e)[0]), literal_eval(e)[1]) for e in labels])  # before 'int', was 'str'
        return labelsDICT

    def parse_file(self, path, ignore_first_row):
        """
        ARGS:
            path: path to file containing a image path and the correspondence class
                for each row. Optionaly it can contain a unique bounding box.
        """
        text = np.array(np.loadtxt(path, str, delimiter='\t'))
        if ignore_first_row:
            text = text[1:]
        return text

    def num_batches(self):
        num_batches = self.num_images / self.batch_size
        # If args are not multiple, we will need one iteration more to cover all samples
        if self.num_images % self.batch_size != 0:
            num_batches += 1
        return num_batches

    def get_object(self):
        return 'images: %i, batches: %i' % (self.num_images, self.num_batches())

    def new_epoch(self):
        self.current_epoch_images = copy.copy(self.images)
        if self.shuffle:
            random.shuffle(self.current_epoch_images)
        self.epoch += 1

    def get_batch(self, batch_size=None):
        # If not batch is specified, use the defined one
        if not batch_size:
            batch_size = self.batch_size

        # Assign the last #batch_size elemenets from self.current_epoch_images and modifies it
        # Note: if batch_size is bigger than the number of elements in the list, return just elements in list
        current_batch, self.current_epoch_images = self.current_epoch_images[-batch_size:], self.current_epoch_images[
                                                                                            :-batch_size]

        # If there are not more images for the current epoch, initialize it with all
        if len(self.current_epoch_images) == 0:
            self.new_epoch()

        return current_batch

    def get_batch_features_spplited(self, batch_size=None):
        """
        Returns:
        """
        batch = self.get_batch(batch_size)
        current_batch = []
        for img_path_class_bbox in batch:
            current_batch.append(self.get_bounding_box(img_path_class_bbox))

        return current_batch

    def get_batch_images(self, size, batch_size=None):
        """
        Returns:
            List of pairs image-classId
        """
        batch = self.get_batch(batch_size)
        current_batch = []
        for img_path_class_bbox_STRING in batch:
            current_batch.append(self.get_bounding_box(img_path_class_bbox_STRING, size, True))

        return current_batch

    # _ at the beggining indicates that it is a private fucntion
    def get_bounding_box(self, image_info, size=None, imageType=False):
        """
        Returns: A dict, with attributes img_path and class. Optionaly it may have bbox attribute
        """
        img_dict = {}
        elements = image_info.split(' ')

        # assing the info we need
        img_dict['class'] = int(elements[1])
        img_dict['class_name'] = self.classesID[int(elements[1])]  # new
        img_dict['name'] = self.get_image_name(elements[0])  # new

        if imageType:
            # get the image
            im = Image.open(elements[0])
            
            if len(elements) == 6:  # there is a bounding box
                im = im.crop([float(elements[2]), float(elements[3]), float(elements[4]), float(elements[5])])
                assert self.multitask == False, "multitask option is not compatible with bounding boxes yet"
                
            elif self.multitask: # multitastk problem
                num_classes = len(self.classesID.keys())
                indices = elements[1:]
                true_labels = []
                for i in range(num_classes):
                    if str(i) in indices:
                        true_labels.append(1)
                    else:
                        true_labels.append(0)
                img_dict['class'] = true_labels
                """
                print 'indices: ', indices, 'type of the first one: ', type(indices[0])
                print true_lables
                """

            im = im.resize(size, Image.LANCZOS)
            img_dict['img'] = np.asarray(im)  # , dtype=np.float16)

        else:
            img_dict['img_path'] = elements[0]
            if len(elements) == 6:  # there is a bounding box
                img_dict['bbox'] = [[elements[2], elements[3]], [elements[4], elements[5]]]

        return img_dict

    def get_image_name(self, img_path):
        return img_path.split(' ')[0].split('/')[-1]
    
    def flip(self, Xb):
        """
        Randomly flips half of the samples
        """
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1, :]
                
        return Xb
    
    def add_median_noise(self, batch, noise=0.3, neighbors=3):
        """
        Randomly applies noise to half of the samples. Median filter is used in order
        to preserve edges.
        """
        bs = batch.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)

        for i in indices:
            batch[i] = batch[i] + noise*batch[i].std()*np.random.random(batch[i].shape)
            batch[i] = ndimage.median_filter(batch[i], neighbors)

        return batch
    
    def rotate_im(self, im, angle=45, verbose=False):
        """rotate the images im
           Args: im is a batch of images with values from 0 to 1
        """
        if verbose:
            print 'type:', type(im), 'shape:', im.shape
            print 'min:', np.min(im), 'max:', np.max(im)
        
        im = Image.fromarray(np.uint8(im)) #im[:,:,0]*255))
        im = im.rotate(angle)#, resample=Image.NEAREST)

        im = np.asarray( im, dtype=np.float32 ) # back to numpy
        
        if verbose:
            print 'type:', type(im), 'shape:', im.shape
            print 'min:', np.min(im), 'max:', np.max(im)
            
        return im
    
    def rotate_images(self, X, max_rotation=50.):
        # choose half of the samples
        bs = X.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)

        for i in indices:
            # choose an angle and rotate
            angle = np.random.uniform(low=-1*max_rotation, high=max_rotation)
            X[i] = self.rotate_im(X[i], angle)

        return X
    
    def transform(self, X):
        """
        Applies a serie of transformations to the current batch
        """
        # copy the objects, so we do not apply changes permanently
        X = X.copy()
        
        # flip half of the samples
        X = self.flip(X)
        
        # Add noise to half of the samples
        #X = self.add_median_noise(X)
        
        # rotate half of the samples
        X = self.rotate_images(X)
        
        return X

    def get_Xy(self, size, batch_size=None):
        """
        Returns:
            Matrix X with the data, vector y with the ground truth
        """
        batch = self.get_batch(batch_size)
        X, y = [], []

        for img_path_class_bbox_STRING in batch:
            img_container = self.get_bounding_box(img_path_class_bbox_STRING, size, True)
            X.append(img_container['img'])
            y.append(img_container['class'])
        #print 'num classes: ', len(self.classesID.keys())
        
        X = np.asarray(X) # to numpy array
        
        if self.data_augmentation:
            print 'Data augmentation activated'
            # Apply transformations. If you do not want to apply, only comment the following line
            X = self.transform(np.asarray(X))
        
        return X, y
    
    def get_Xy_name(self, size, batch_size=None):
        """
        Returns:
            Matrix X with the data, vector y with the ground truth
        """
        batch = self.get_batch(batch_size)
        X, y, names = [], [], []

        for img_path_class_bbox_STRING in batch:
            img_container = self.get_bounding_box(img_path_class_bbox_STRING, size, True)
            X.append(img_container['img'])
            y.append(img_container['class'])
            names.append(img_path_class_bbox_STRING)
        #print 'num classes: ', len(self.classesID.keys())
        
        X = np.asarray(X) # to numpy array
        
        if self.data_augmentation:
            print 'Data augmentation activated'
            # Apply transformations. If you do not want to apply, only comment the following line
            X = self.transform(np.asarray(X))
        
        return X, y, names

    def get_num_images(self):
        return self.num_images

def test():
    print '\nTEST'
    N = 368
    B = 23
    file_name = 'data/testingObjects/test.txt'
    class_id_files = 'data/testingObjects/classesID.txt'
    dSet_train = Dataset(file_name, N, B, class_id_files)
    image_name = dSet_train.get_image_name(dSet_train.images[0])
    print 'image folder-name: ', image_name
    dSet_train.get_object()
    print 'num batches: ', dSet_train.num_batches()
    epochs = 3

    imagesCount = 0
    for epoch in range(epochs):

        images = []
        print 'epoch: ', dSet_train.epoch, 'images viewed: ', imagesCount

        for i in range(dSet_train.num_batches()):
            current_batch = dSet_train.get_batch()
            images.append(current_batch)
            imagesCount += len(current_batch)
            print 'batch_lenght; ', len(current_batch)
            print len(dSet_train.current_epoch_images) - dSet_train.num_images
    print imagesCount == N * epochs
    

def testBoundingBox():
    print '\n TEST BOUNDINX BOXES'
    N = 6
    B = 4
    file_name = 'data/testingObjects/test_bouning.txt'
    class_id_files = 'data/testingObjects/classesID.txt'
    ignore_first_row = True
    dSet_train = Dataset(file_name, N, B, class_id_files, ignore_first_row)
    dSet_train.get_object()
    print 'num batches: ', dSet_train.num_batches()
    print dSet_train.images
    current_batch = dSet_train.get_batch_features_spplited()
    print current_batch


def testBOX():
    print '\n TEST BOUNDINX BOXES'
    N = 2
    B = 4
    size = [256, 256]
    file_name = 'data/testingObjects/test_box.txt'
    class_id_files = 'data/testingObjects/classesID.txt'
    ignore_first_row = True
    dSet_train = Dataset(file_name, N, B, class_id_files, ignore_first_row)
    dSet_train.get_object()
    print dSet_train.classesID
    print 'num batches: ', dSet_train.num_batches()
    print dSet_train.images
    current_batch = dSet_train.get_batch_images(size)
    print 'num elements: ', len(current_batch), 'num attributes: ', len(current_batch[0])
    plt.imshow(current_batch[0]['img'])
    plt.show()


if __name__ == '__main__':
    test()
    testBoundingBox()
    testBOX()
"""
import numpy as np

nclasses = 6
A = [1,2,3,4,5,5,5,4,2,0]
lA = len(A)

B = np.zeros((lA,nclasses))
B[range(lA),A]=1

print B
"""

GPU = True

if GPU:
    # use keras and tensorflow limitating gpu memory
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.48
    set_session(tf.Session(config=config))
else:
    # only cpu
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


from cnn.inceptionV3.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras import backend as K

# general imports
import logging
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import itertools

# import our scripts
SCRIPTS_PATH = '/../../scripts'
sys.path.insert(0, os.getcwd()+SCRIPTS_PATH)
#print sys.path
from dataset import Dataset


##### MODEL SPECIFICATIONS - CHANGE THIS FOR EACH DIFFERENT EXPERIMENT
#All the dataset for FOOD
TITLE = 'Food' 
DATASET_FOLDER = '../../data/datasets/food_dataset'
WEIGHTS_NAME = 'weights/weights_35-34_71-15_74_6536.h5' # to load pretrained weights
NUM_CLASSES = 3
NUM_TRAIN_IMG = 5582 # just to check if the dataset was correctly loaded
NUM_VAL_IMG = 1196   # just to check if the dataset was correctly loaded
NUM_TEST_IMG = 1195   # just to check if the dataset was correctly loaded

BATCH_SIZE = 32      # Depends on the available GPU memory
MULTITASK = False
CLASS_WEIGHT = {}

"""

#All the dataset for FOOD
TITLE = 'Table' 
DATASET_FOLDER = '../../data/datasets/table_dataset'
WEIGHTS_NAME = 'weights/weights_85-30_85-45_25_1040.h5'
NUM_CLASSES = 2
NUM_TRAIN_IMG = 5582 # just to check if the dataset was correctly loaded
NUM_VAL_IMG = 1196   # just to check if the dataset was correctly loaded
NUM_TEST_IMG = 1195   # just to check if the dataset was correctly loaded

BATCH_SIZE = 32      # Depends on the available GPU memory
MULTITASK = False
CLASS_WEIGHT = {}
"""

"""
# All the Dataset with the 12 possible classes
TITLE = 'full_dataset_new' 
DATASET_FOLDER = '../../data/datasets/full_dataset'
WEIGHTS_NAME = 'weights/weights_14-44_19-15_e3_i1983_full_dataset.h5'
NUM_CLASSES = 12
NUM_TRAIN_IMG = 31708 # just to check if the dataset was correctly loaded
NUM_VAL_IMG = 6795   # just to check if the dataset was correctly loaded
NUM_TEST_IMG = 6794  # just to check if the dataset was correctly loaded
BATCH_SIZE = 16     # Depends on the available GPU memory
CLASS_WEIGHT={}
#CLASS_WEIGHT = {0:1.0, 1:21106/5877, 2:21106/9755, 3:21106/5194, 4:21106/187, 5:21106/218, 6:21106/425, 7:21106/673, 8:21106/88, 9:21106/543, 10:21106/242, 11:21106/1037}
MULTITASK = False

"""
"""
#All the dataset multitask
TITLE = 'multitask_weighted_all' 
DATASET_FOLDER = '../../data/datasets/multitask_dataset'
WEIGHTS_NAME = 'weights/weights_70-02_74-73_45-28_e8_i8918_multitask_weighted_15_corrected_2.h5'
#WEIGHTS_NAME = 'weights/weights_75-40_75-40_e4_i2479_multitask_weights.h5'
NUM_CLASSES = 7
NUM_TRAIN_IMG = 31708 # just to check if the dataset was correctly loaded
NUM_VAL_IMG = 6795   # just to check if the dataset was correctly loaded
NUM_TEST_IMG = 6794  # just to check if the dataset was correctly loaded
BATCH_SIZE = 64      # Depends on the available GPU memory
MULTITASK = True
MULTITASK_LBL = ["Food", "Social", "Table"]
#CLASS_WEIGHT = {}
CLASS_WEIGHT = {0:1.0, 1:92.47/3.32 , 2: 92.47/4.22 , 3:1.0 ,4: 61.75/38.25, 5: 1.0,6:70.1/29.9 }
"""

"""
#Part of the dataset multitask
#train multitask
TITLE = 'multitask_test' 
DATASET_FOLDER = '../../data/datasets/Test_multitask'
WEIGHTS_NAME = 'weights/weights_87-57_87-57_e66_i5895_multitask.h5'
NUM_CLASSES = 7
NUM_TRAIN_IMG = 100 # just to check if the dataset was correctly loaded
NUM_VAL_IMG = 100   # just to check if the dataset was correctly loaded
NUM_TEST_IMG = 100   # just to check if the dataset was correctly loaded

BATCH_SIZE = 32      # Depends on the available GPU memory
MULTITASK = True
MULTITASK_LBL = ["Food", "Table", "Social"]
CLASS_WEIGHT = {0:1.0, 1:95.67/1.78 ,2: 95.67/2.55 , 3:1.0 ,4: 81.31/18.69,5: 1.0,6:74.95/25.05 }
"""
"""
#All the dataset multitask
TITLE = 'multitask' 
DATASET_FOLDER = '../../data/datasets/multitask_dataset'
WEIGHTS_NAME = 'weights/weights_85-30_85-45_25_1040.h5'
NUM_CLASSES = 7
NUM_TRAIN_IMG = 5582 # just to check if the dataset was correctly loaded
NUM_VAL_IMG = 1196   # just to check if the dataset was correctly loaded
NUM_TEST_IMG = 1195   # just to check if the dataset was correctly loaded

BATCH_SIZE = 64      # Depends on the available GPU memory
MULTITASK = True
MULTITASK_LBL = ["Food", "Table", "Social"]
CLASS_WEIGHT = {0:1.0, 1:95.67/1.78 ,2: 95.67/2.55 , 3:1.0 ,4: 81.31/18.69,5: 1.0,6:74.95/25.05 }
"""

"""
#All the dataset for SOCIAL
TITLE = 'Social' 
DATASET_FOLDER = '../../data/datasets/company_dataset'
WEIGHTS_NAME = 'weights/weights_85-30_85-45_25_1040.h5'
NUM_CLASSES = 2
NUM_TRAIN_IMG = 5582 # just to check if the dataset was correctly loaded
NUM_VAL_IMG = 1196   # just to check if the dataset was correctly loaded
NUM_TEST_IMG = 1195   # just to check if the dataset was correctly loaded

BATCH_SIZE = 32      # Depends on the available GPU memory
MULTITASK = False
CLASS_WEIGHT = {}
"""
"""

##### MODEL TEST SPECIFICATIONS 
TITLE = 'test_dataset' 
DATASET_FOLDER = '../../data/datasets/Test'
WEIGHTS_NAME = 'weights/weights_28-97_64-30_5_516.h5'
NUM_CLASSES = 12
NUM_TRAIN_IMG = 122 # just to check if the dataset was correctly loaded
NUM_VAL_IMG = 74    # just to check if the dataset was correctly loaded
NUM_TEST_IMG = 100  # just to check if the dataset was correctly loaded
BATCH_SIZE = 16   # Depends on the available GPU memory
CLASS_WEIGHT = {}
MULTITASK = False
"""

##### GENERAL SETTINGS
TRAIN_TXT = DATASET_FOLDER+'/train.txt'
VAL_TXT = DATASET_FOLDER+'/val.txt'
TEST_TXT=DATASET_FOLDER+'/test.txt'
CLASSES = DATASET_FOLDER+'/classesID.txt'
OUTPUTS = 'outputs/'
WEIGHTS_PATH = OUTPUTS + WEIGHTS_NAME
NAME_SAVED_MODEL = OUTPUTS + 'weights/weights_'
CONF_MATRIX = OUTPUTS + 'confusion_matrices/cm_'
ACCU_LOSS = OUTPUTS + 'accuracy_loss/acc_loss_'
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
SIZE = [MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT]
#EVAL_INTERVAL = NUM_TRAIN_IMG #int( (1.0*NUM_TRAIN_IMG)/BATCH_SIZE/2 )  # every 1/2 of all images

# set the log
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S')

# if you set FLAG_DEBUG = True, you will not run the whole experiment. It will just
# run few samples
FLAG_DEBUG = False

if FLAG_DEBUG:
    EVAL_INTERVAL = 1


def plot_batch(images, titles=None, correct=None, num_images_row = 7, image_width=2):

    num_rows = int(np.ceil(1.*len(images)/num_images_row))
    
    fig = plt.figure(figsize = (image_width*num_images_row,1.*13/14*image_width*num_rows))
    gs1 = gridspec.GridSpec(num_rows, num_images_row)
    gs1.update(wspace=0.25, hspace=0.25)

    
    for i, img in enumerate(images):
        ax = plt.subplot(gs1[i])
        
        if titles:
            if correct:
                if correct[i]: # if was correct
                    col = 'g'
                else:
                    col = 'r'
                ax.set_title(str(i)+' '+titles[i], color=col)
            else:
                ax.set_title(str(i)+' '+titles[i])
        else:
            ax.set_title(str(i))
            
        ax.imshow(img, aspect=1)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Avoid unwanted white spaces while sharing axes
        ax.set_adjustable('box-forced')

    fig.canvas.draw()
    return fig

def save_plot(loss, acc, norm_acc, name_to_save):
    # transform arrays to numpy arrays
    acc = np.asarray(acc)
    norm_acc = np.asarray(norm_acc)

    # prepare the plot
    fig = plt.figure()
    plt.plot(loss, 'ro-', label='Loss')
    plt.plot(acc[:, 0], acc[:, 1], 'bo--', label='Acc')
    plt.plot(norm_acc[:, 0], norm_acc[:, 1], 'go--', label='Nor Acc')
    plt.legend(loc='center left')

    # save the figure and close so it never gets displayed
    plt.savefig(ACCU_LOSS + name_to_save + '.png')
    plt.close(fig)

def save_confusion_matrix(y_true, y_pred, name_to_save, title='Confusion matrix', size_cm=NUM_CLASSES, acc='', acc_norm=''):
    # create the confusion matrix and normalize it
    cm = 1.0*confusion_matrix(y_true, y_pred, labels=range(size_cm))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    inf = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # prepare the plot
    fig = plt.figure()
    plt.matshow(cm_norm)
    tick_marks = np.arange(cm_norm.shape[0])
    plt.xticks(tick_marks, rotation=45)
    plt.yticks(tick_marks)
    if (NUM_CLASSES!=12):
        for i, j in itertools.product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
            plt.text(j, i, str(int(cm[i,j]))+' / '+str(int(sum(cm[i]))),
                     horizontalalignment="center",
                     color="white" if cm_norm[i, j] < 0.5 else "black")

    plt.title(title)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label \n '+ 'Acc.: '+str(round(acc,2))+ ' Acc. norm.: '+str(round(acc_norm,2))+'\n F1 score: '+str(round(inf[2],2))+'\n Precision: '+str(round(inf[0], 2))+', Recall: '+str(round(inf[1], 2)))

    # save the figure and close so it never gets displayed
    plt.savefig(CONF_MATRIX+name_to_save+'.png')
    plt.close(fig)

def save_checkpoint(model, name, loss, val_accuracy, val_accuracy_normalized, y_true, y_pred, acc, acc_norm):
    # save the confusion matrix and the plot of it
    if MULTITASK:
        y_true, len_cm = create_vector_confusion_matrix(y_true)
        y_pred, _ = create_vector_confusion_matrix(y_pred)
        for i, lbl in enumerate(MULTITASK_LBL):
            save_confusion_matrix(y_true[i], y_pred[i], name+lbl, size_cm=len_cm[i], acc=acc[i], acc_norm=acc_norm[i])
    else:
        save_confusion_matrix(y_true, y_pred, name, acc=acc, acc_norm=acc_norm)
    # save the picture of accuracy and loss
    save_plot(loss, val_accuracy, val_accuracy_normalized, name)

    # save the learned weights so far
    model.save_weights(NAME_SAVED_MODEL+name+'.h5')

#Passing a solution of a matrix with binary solutions (ex. [[1,0,0], [0,0,1]]) passing it to a single vector not binary (ex. [0,2])
def create_vector_confusion_matrix(y):
    y_cm = [[], [], []]
    len_cm = [3, 2, 2]
    y_cm[0] = np.asarray([lbl[0:3] for lbl in y]).argmax(1)
    y_cm[1] = np.asarray([lbl[3:5] for lbl in y]).argmax(1)
    y_cm[2] = np.asarray([lbl[5:7] for lbl in y]).argmax(1)
    return y_cm, len_cm
                                  
def preprocess_input(x):
    x = x/255.
    x = x - 0.5
    x = x*2.
    return x

def evaluate(model, val_ds):
    validation_accuracy = 0.
    # Initialize the array to save the accuracy of all our examplesw
    acc_norm_top1 = [[] for i in xrange(NUM_CLASSES)]
    # Initialize the arrays for the Confusion Matrix
    y_true = []
    y_pred = []

    if FLAG_DEBUG:
        num_validations = 10
    else:
        num_validations = val_ds.num_batches()

    for iter_val in range(num_validations):
        print 'iter_val: %d/%d' %( iter_val+1, val_ds.num_batches() )
        # Get new batch and preprocess the image according to the model
        X, y = val_ds.get_Xy(SIZE)
        X = preprocess_input(X)

        # Feed batches to your model manually, and get the accuracy
        out = model.predict(X)
        validation_accuracy += np.sum(out.argmax(1) == y)

        # Update the normalized accuracy and elements to compute the confusion matrix
        for pred, lbl in zip(out.argmax(1), y):
            acc_norm_top1[int(lbl)].append(float(pred == lbl))
            y_true.append(lbl)
            y_pred.append(pred)

    # compute the total accuracies and print
    validation_accuracy = validation_accuracy / val_ds.get_num_images()
    num_classes_with_predictions = sum([1 for e in acc_norm_top1 if e])
    f1 = f1_score(y_true, y_pred,average='weighted') 
    if num_classes_with_predictions > 0:
        norm_acc = sum([sum(classAccur) / len(classAccur) for classAccur in acc_norm_top1 if
             len(classAccur) > 0]) / num_classes_with_predictions
    else:
        print "Something is wrong during evaluation, num_classes_with_predictions=", \
            num_classes_with_predictions
        sys.exit(1)
    info_cm = 'Accuracy: '+str(round(validation_accuracy, 2))+'\n Normalized accuracy: '+str(round(norm_acc, 2))+'\n F1 score: '+str(round(f1, 2))
    print 'Validation Accuracy: %f' % (validation_accuracy)
    print 'Validation Normalized Accuracy: %f' % (norm_acc)

    return validation_accuracy, norm_acc, y_pred, y_true, validation_accuracy,norm_acc

def evaluate_multitask(model, val_ds):
    # Initialize the arrays for the Confusion Matrix
    y_true = []
    y_pred = []

    if FLAG_DEBUG:
        num_validations = 10
    else:
        num_validations = val_ds.num_batches()

    for iter_val in range(num_validations):
        print 'iter_val: %d/%d' %( iter_val+1, num_validations)
        # Get new batch and preprocess the image according to the model
        X, y = val_ds.get_Xy(SIZE)
        X = preprocess_input(X)

        # Feed batches to your model manually, and get the accuracy
        out = model.predict(X)
        # Update the normalized accuracy and elements to compute the confusion matrix
        for (el, lbl) in zip(out, y):
            pred = [0,0,0,0,0,0,0]
            pred[el[:3].argmax()] = 1
            pred[el[3:5].argmax()+3] = 1
            pred[el[5:7].argmax()+5] = 1

            y_pred.append(pred)
            y_true.append(lbl)
            
    # compute the total accuracies and print
    validation_accuracy = accuracy(y_true, np.asarray(y_pred))
    
    #Compute the accuracy of all the image (if all the labels are correct or not)
    validation_accuracy2 = accuracy2(y_true, np.asarray(y_pred))
    print "Validation accuracy 2 (completely true)", validation_accuracy2
    
    acc_norm_food = normalized_accuracy(to_one_vector(y_true, [0,3]), to_one_vector(y_pred, [0,3]))
    acc_norm_table = normalized_accuracy(to_one_vector(y_true, [3,5]), to_one_vector(y_pred, [3,5]))
    acc_norm_social = normalized_accuracy( to_one_vector(y_true, [5,7]), to_one_vector(y_pred, [5,7]))
    acc_norm = [acc_norm_food, acc_norm_table, acc_norm_social]
    
    #compute the accuracy of each class
    food_acc = accuracy2(np.asarray([lbl[0:3] for lbl in y_true]), np.asarray([lbl[0:3] for lbl in y_pred]))
    table_acc =accuracy2(np.asarray([lbl[3:5] for lbl in y_true]), np.asarray([lbl[3:5] for lbl in y_pred]))
    social_acc = accuracy2(np.asarray([lbl[5:7] for lbl in y_true]), np.asarray([lbl[5:7] for lbl in y_pred]))
    #save the information to print in the confusion matrix
    acc = [food_acc, table_acc, social_acc]
    print 'Validation Accuracy: %f' % (validation_accuracy)
    #print 'Validation Normalized Accuracy: %f' % (norm_acc)
    print "Validation accuracy 2 (completely true)", validation_accuracy2
    return validation_accuracy, sum(acc_norm)/len(acc_norm), y_pred, y_true, acc, acc_norm, validation_accuracy2

def normalized_accuracy(y_true, y_pred):
    cm = 1.0*confusion_matrix(y_true, y_pred, labels=range(len(set(y_true))))
    acc = []
    for i, row in enumerate(cm):
        acc.append(row[i]/sum(row))
    return sum(acc)/len(acc)

def to_one_vector(y, pos):
    y = np.asarray(y)
    return np.asarray([lbl[pos[0]:pos[1]].argmax() for lbl in y])    


# Accuracy of "each class" of an image. If 2 of the 3 classes are predicted good: 2/3 acc
def accuracy(y_true, y_pred):
    acc = 0.0
    for true, pred in zip(y_true, y_pred):
        acc += (sum(true==pred)-1)/float(len(true)-1)
    return acc/float(len(y_true))


#Accuracy 1 if all the classes of an image are predicted true, else 0
def accuracy2(y_true, y_pred):
    acc = 0.0
    for true, pred in zip(y_true, y_pred):
        if(sum(true==pred)==len(true)): acc+=1
    return acc/len(y_true)
  
    
def train(model, train_ds, val_ds, n_epochs, verbose=False):
    loss_training = []
    val_accuracy = []
    val_accuracy_normalized = []

    if FLAG_DEBUG:
        num_iterations = 50
    else:
        num_iterations = train_ds.num_batches() * n_epochs
    
    # Run the training as many cycles requested.
    for i in range(num_iterations):
        print '\n######################################## Epoch %d/%d, \
    Batch %d/%d' % (train_ds.epoch+1, n_epochs, i%train_ds.num_batches()+1, 
                       train_ds.num_batches())

        # Get new batch already preprocess
        X, y = train_ds.get_Xy(SIZE)
        if verbose:
            plot_batch(X)

        X = preprocess_input(X)
        
        # Feed batches to your model manually, train and save the loss
        train_loss = model.train_on_batch(X, y, class_weight = CLASS_WEIGHT )
        print train_loss
        loss_training.append(train_loss)
        print "Mini-Batch loss: ", train_loss

        # Every so often, print out how well the model is training
        is_last_step = (i + 1 == num_iterations)
        if ((((i+1) % train_ds.num_batches()) == 0 or is_last_step) and i!=0):
            print '\n######################################## Validation:', (i+1)/train_ds.num_batches()
            #evaluate the validation set, and save validation accuracies
            if MULTITASK:
                current_val_acc, current_val_acc_norm, y_pred, y_true, acc, acc_norm, val_acc_complete = evaluate_multitask(model,val_ds)
            else:
                current_val_acc, current_val_acc_norm, y_pred, y_true, acc, acc_norm = evaluate(model,val_ds)
            val_accuracy.append([i, current_val_acc])
            val_accuracy_normalized.append([i, current_val_acc_norm])
            # save the learned model so far
            if MULTITASK:
                name = "{0:.2f}".format(current_val_acc_norm * 100).replace('.', '-') + '_' +\
                       "{0:.2f}".format(current_val_acc * 100).replace('.', '-') + '_' +\
                    "{0:.2f}".format(val_acc_complete * 100).replace('.', '-') + '_e' + \
                       str(train_ds.epoch-1) + '_i' + str(i) + '_' + TITLE
            else:
                name = "{0:.2f}".format(current_val_acc_norm * 100).replace('.', '-') + '_' +\
                       "{0:.2f}".format(current_val_acc * 100).replace('.', '-') + '_e' + \
                       str(train_ds.epoch-1) + '_i' + str(i) + '_' + TITLE
            save_checkpoint(model, name, loss_training, val_accuracy, val_accuracy_normalized, y_true, y_pred, acc, acc_norm)
            
            
def freeze_n_first_layers(n_not_train, model, optimizer=SGD(lr=0.001, momentum=0.9),
                          loss='sparse_categorical_crossentropy'):

    for layer in model.layers[:n_not_train]:
        layer.trainable = False
    for layer in model.layers[n_not_train:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=optimizer, loss=loss)

    
def visualize_model(model):
    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    print 'Model topology:\n'
    for i, layer in enumerate(model.layers):
        print(i, layer.name)

        
def buildModel_egocentric(num_old_classes=2, load_weights = False):
    # create the base pre-trained model
    if load_weights == False:
        base_model = InceptionV3(weights='imagenet', include_top=False)
    else:
        base_model = InceptionV3(weights=None, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer with 2 classes food/no_food
    predictions = Dense(num_old_classes, activation='softmax')(x)

    # this is the model we will train
    print 'Loading model...\n'
    model = Model(input=base_model.input, output=predictions)

    if load_weights:
        model.load_weights(WEIGHTS_PATH)

    return model


def buildModel(load_weights = False):
    
    # create the base pre-trained model
    if load_weights == False:
        base_model = InceptionV3(weights='imagenet', include_top=False)
    else:
        base_model = InceptionV3(weights=None, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer with NUM_CLASSES classes food/no_food
    predictions = Dense(NUM_CLASSES, activation='sigmoid')(x)

    # this is the model we will train
    print 'Loading model...\n'
    model = Model(input=base_model.input, output=predictions)

    if load_weights:
        model.load_weights(WEIGHTS_PATH)

    return model


def adaptModel(model):
    n_specialist_coordinates = NUM_CLASSES

    # Get input
    new_input = model.input

    # Find the layer to connect
    hidden_layer = model.layers[-2].output

    # Connect a new layer on it
    new_output = Dense(n_specialist_coordinates, activation='sigmoid')(hidden_layer)

    # Build a new model
    model = Model(new_input, new_output)

    return model




def main():
    """"invoke this to train earlier layers."""

    # ------------------- Data
    # prepare the training and validation dataset
    train_ds = Dataset(TRAIN_TXT, NUM_TRAIN_IMG, BATCH_SIZE, CLASSES, multitask=MULTITASK)
    val_ds = Dataset(VAL_TXT, NUM_VAL_IMG, BATCH_SIZE, CLASSES, multitask=MULTITASK,
                     data_augmentation=False)
    test_ds = Dataset(TEST_TXT, NUM_TEST_IMG, BATCH_SIZE, CLASSES, 
                      data_augmentation=False, multitask = MULTITASK)
    
    # ------------------- Model from egocentric training
    # load the desired model and visualize it
    model = buildModel(load_weights=True)
    # visualize_model(model) #old
    #model.summary()

    # ------------------- Model from egocentric
    # load the desired model and visualize it
    #model = adaptModel(model)
    #model = buildModel(True)
    # ------------------- Train more Layers
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    if (MULTITASK):
        freeze_n_first_layers(10, model,loss = 'binary_crossentropy')
    else:
        freeze_n_first_layers(-2, model, SGD(lr=0.001, momentum=0.9))
        
    
    # visualize_model(model) # once freezed almost all layers
    model.summary()
    
    epoch_top_layers = 100
    train(model, train_ds, val_ds, epoch_top_layers)

def main2():
    # ------------------- Data
    # prepare the training and validation dataset
    train_ds = Dataset(TRAIN_TXT, NUM_TRAIN_IMG, BATCH_SIZE, CLASSES)
    val_ds = Dataset(VAL_TXT, NUM_VAL_IMG, BATCH_SIZE, CLASSES)
    test_ds = Dataset(TEST_TXT, NUM_TEST_IMG, BATCH_SIZE, CLASSES, 
                      data_augmentation=False, multitask = MULTITASK)
    
    # ------------------- Model
    # load the desired model and visualize it
    model = buildModel()   
    #visualize_model(model) #old
    model.summary()

    # ------------------- Train just new Layers
    # first: train only the top layers (which were z initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    freeze_n_first_layers(-3, model, loss = 'binary_crossentropy')

    # train the model on the new data for a few epochs
    epoch_top_layers = 10
    train(model, train_ds, val_ds, epoch_top_layers)
    # At this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # ------------------- Train more Layers
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    #freeze_n_first_layers(172, model, SGD(lr=0.0001, momentum=0.9))

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    epoch_top_layers = 10
    #train(model, train_ds, val_ds, epoch_top_layers)

if __name__ == '__main__':
    main()

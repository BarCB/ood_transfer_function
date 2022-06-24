import torchvision
import os
from sklearn.model_selection import train_test_split
import ntpath
from shutil import copy2
import random
import numpy as np

def create_train_test_folder_partitions(datasetpath_base, percentage_evaluation=0.25, random_state=42, batch=0, create_dirs = True, classes_in_dist = []):
    """
    Train and test partitioner
    :param datasetpath_base:
    :param percentage_used_labeled_observations: The percentage of the labeled observations to use from the 1 -  percentage_evaluation
    :param num_batches: total number of batches
    :param create_dirs:
    :param percentage_evaluation:  test percentage of dat
    :return:
    """

    #for the same batch, same result
    random.seed(random_state + batch)

    datasetpath_all = os.path.join(datasetpath_base, "all")
    print("All data  path: ", datasetpath_all)
    #read dataset
    dataset = torchvision.datasets.ImageFolder(datasetpath_all)
    #get filenames
    list_file_names_and_labels = dataset.imgs
    labels_temp = dataset.targets
    list_file_names = []
    list_labels = []
    list_files_labels = []
    # list of file names and labels
    for i in range(0, len(list_file_names_and_labels)):
        file_name_path = list_file_names_and_labels[i][0]
        list_file_names += [file_name_path]
        list_labels += [labels_temp[i]]
        #for random swapping
        list_files_labels += [(file_name_path, labels_temp[i])]
    #total number of classes
    num_classes = len(np.unique(np.array(labels_temp)))
    print("Total number of classes detected: ", num_classes)
    #if no custom in distribution classes were chosen, take them all
    if(classes_in_dist == []):
        print("No out of distribution data required")
        for i in range(0, num_classes): classes_in_dist += [i]

    #get the number of observations for the less represented class
    array_labels = np.array(list_labels)
    observations_under_class = len(array_labels[array_labels == 0])
    array_labels = np.array(list_labels)
    #find the under represented class
    for i in range(0, num_classes):
        num_obs_class = len(array_labels[array_labels == i])
        if(num_obs_class < observations_under_class):
            observations_under_class = num_obs_class
    print("Under represented class with the following number of observations ", observations_under_class)
    datasetpath_train = os.path.join(datasetpath_base, "batches_labeled", "batch_" + str(batch),"train")
    datasetpath_test = os.path.join(datasetpath_base, "batches_labeled", "batch_" + str(batch), "test")
    if (create_dirs):
        # create the directories
        for i in classes_in_dist:
            print("Creating test path: ", datasetpath_test + "/" + str(i))
            print("Creating training path: ", datasetpath_train + "/" + str(i))
            try:
                os.makedirs(os.path.join(datasetpath_train, str(i)))
            except:
                print("Rewritting directories...")
            try:
                os.makedirs(os.path.join(datasetpath_test, str(i)))
            except:
                print("Rewritting directories...")

    # test and train  splitter for unlabeled and labeled data split
    #for the same batch number, same results

    for curr_class in range(0, num_classes):
        #get all the labels of the

        (list_files_class, list_labels_class) = undersample_list(observations_under_class, list_files_labels, curr_class)
        print("Undersampled list of files size ", len(list_files_class), " for class ", curr_class)
        print("Undersampled list of labels size ", len(list_labels_class), " for class ", curr_class)

        X_train, X_test, y_train, y_test = train_test_split(list_files_class, list_labels_class, test_size=percentage_evaluation, random_state=random_state + batch)
        print("Creating trainig partitioned folders...", len(X_train))
        for i in range(0, len(X_train)):
            path_src = X_train[i]
            # extract the file name
            file_name = ntpath.basename(path_src)
            label = y_train[i]
            path_dest = os.path.join(datasetpath_train, str(label), file_name)
            #copy2(path_src, path_dest)

        print("Creating test partitioned folders...", len(X_test))
        for i in range(0, len(X_test)):
            label = y_test[i]
            # if the label is among the in distribution selected label, copy it there
            if (label in classes_in_dist):
                path_src = X_test[i]
                file_name = ntpath.basename(path_src)
                path_dest = os.path.join(datasetpath_test, str(y_test[i]), file_name)
                #copy2(path_src, path_dest)

def undersample_list(observations_under_class, list_files_labels, curr_class):
    random.shuffle(list_files_labels)
    #load only the labels
    array_labels = np.array([element[1] for element in list_files_labels])


    list_file_indices_class = (array_labels == curr_class)
    list_files_class_selected = []
    list_labels_class_selected = []
    # get the file names of the class
    number_added = 0
    # undersample the folders to the lowest num of observations per class
    for index in range(0, len(list_file_indices_class)):
        if (list_file_indices_class[index] and number_added < observations_under_class):
            list_files_class_selected += [list_files_labels[index][0]]
            list_labels_class_selected += [list_files_labels[index][1]]
            number_added += 1


    print("Number of labels undersampled ", len(list_labels_class_selected))
    return (list_files_class_selected, list_labels_class_selected)


for i in range(5, 10):
    create_train_test_folder_partitions("C:\\Users\\Barnum\\Desktop\\datasets\\labelled\\MNIST", batch=i)
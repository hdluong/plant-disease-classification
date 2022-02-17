import os

def load_train_sets(train_path, valid_path):
    train_data = []
    train_label = []
    valid_data = []
    valid_label = []
    classes = len(os.listdir(train_path))

    print("[INFO]: Load train data ...")
    # train data
    for folder in os.listdir(train_path):
        for file in os.listdir(os.path.join(train_path, folder)):
            file_path = os.path.join(train_path, folder, file)
            train_data.append(file_path)
            train_label.append(folder)

    # validation data
    for folder in os.listdir(valid_path):
        for file in os.listdir(os.path.join(valid_path, folder)):
            file_path = os.path.join(valid_path, folder, file)
            valid_data.append(file_path)
            valid_label.append(folder)

    print("[INFO]: Load train data done!")
    
    return train_data, train_label, valid_data, valid_label, classes

def load_test_set(test_path):
    test_data = []
    test_label = []

    print("[INFO]: Load test data ...")
    # test data
    for folder in os.listdir(test_path):
        for file in os.listdir(os.path.join(test_path, folder)):
            file_path = os.path.join(test_path, folder, file)
            test_data.append(file_path)
            test_label.append(folder)
    print("[INFO]: Load test data done!")

    return test_data, test_label
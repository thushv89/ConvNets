

# Makes the data reading easy
# for imagenet-100 data needs to be read from memmaps
# for cifar data needs to be read from pickles
# encapsulates this type of changes

def read_cifar_10(datafile):

    train_dataset,train_labels = None,None
    valid_dataset,valid_labels = None,None
    test_dataset,test_labels = None,None


def read_imagenet_100(train_datafile,valid_datafile):
    
import os

def remove_and_create_dir(path):
    """ System call to rm -rf and then re-create a dir """

    dir = os.path.dirname(path)
    print('attempting to delete ', dir, ' path ', path)
    if os.path.exists(path):
        os.system("rm -rf " + path)
    os.system("mkdir -p " + path)

    return path

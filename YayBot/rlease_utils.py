'''
Utilities for various scripts
'''
import pickle
from os.path import expanduser
from pathlib import Path


def get_save_path_and_make_save_directory(file_name, file_dir = "/rlease_results/"):
    """
    Makes directory and returns save path
    :param file_name: file name to save to
    :param file_dir: directory to put file of file name in
    :return: a file path represented bya  string
    """
    save_dir = expanduser('~') + file_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = save_dir + file_name
    return save_path


def pickle_file(file_name, save_object):
    with open(file_name, 'wb') as handle:
        pickle.dump(save_object, handle, protocol=pickle.HIGHEST_PROTOCOL)




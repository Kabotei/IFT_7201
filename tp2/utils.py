from pathlib import Path
import json
import pickle
import numpy as np


def create_dirs(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def create_parent_dirs(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_dict_json(path: str) -> dict:
    with open(path, 'r') as file:
        return json.load(file)


def save_dict_json(path: str, data: dict):
    with open(path, 'w') as file:
        json.dump(data, file)


def load_dict_ser(path: str) -> object:
    with open(path, 'rb') as file:
        return pickle.load(file)


def save_dict_ser(path: str, data: object):
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

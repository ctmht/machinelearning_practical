import pickle as pkl


def save(instance, path: str):
    with open(path, 'wb') as file:
        pkl.dump(instance, file)


def load(filename):
    with open(filename, 'rb') as file:
        instance = pkl.load(file)
    return instance

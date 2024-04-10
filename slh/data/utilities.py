import numpy as np


def split_idxs(dataset_length: int, n_train: int, n_valid: int):
    idxs = np.arange(dataset_length)
    np.random.shuffle(idxs)
    train_idxs = idxs[:n_train]
    val_idxs = idxs[n_train : n_train + n_valid]

    return train_idxs, val_idxs


def split_dataset(dataset_list, train_idxs, val_idxs=None):
    """Shuffles and splits a list in two resulting lists
    of the length length1 and length2.

    Parameters
    ----------
    data_list :
        A list.
    length1 :
        Length of the first resulting list.
    length2 :
        Length of the second resulting list.

    Returns
    -------
    splitted_list1
        List of random structures from atoms_list of the length length1.
    splitted_list2
        List of random structures from atoms_list of the length length2.
    """
    train_ds_list = [dataset_list[i] for i in train_idxs]

    if val_idxs is not None:
        val_ds_list = [dataset_list[i] for i in val_idxs]
    else:
        val_ds_list = []

    return train_ds_list, val_ds_list

import torchutils
import numpy as np


def generate_data(num_users, num_tuple, num_items, num_cates, max_size):
    data = []
    for u in range(num_users):
        for i in range(num_tuple):
            size = np.random.randint(low=2, high=max_size + 1, size=1)[0]
            append = [-1] * (max_size - size)
            items = np.random.randint(low=0, high=num_items, size=size)
            cates = np.random.randint(low=0, high=num_cates, size=size)
            data.append([u, size] + items.tolist() + append + cates.tolist() + append)
    return np.array(data)


def test_outfit():
    num_users = 10
    num_tuple = 300
    num_items = 50
    num_cates = 10
    max_size = 8
    data = generate_data(num_users, num_tuple, num_items, num_cates, max_size)
    outfit = torchutils.fashion.OutfitTuple(data)
    assert outfit.num_user == num_users
    assert outfit.num_type <= 10
    assert outfit.max_size <= max_size

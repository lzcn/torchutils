import numpy as np
import torchutils


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


def test_fix_generator():
    num_users = 10
    num_tuple = 300
    num_items = 50
    num_cates = 10
    max_size = 8
    data = generate_data(num_users, num_tuple, num_items, num_cates, max_size)
    generator = torchutils.fashion.getGenerator(mode="Fix", data=data)
    assert (generator(data) == data).all()


def test_identity():
    num_users = 10
    num_tuple = 300
    num_items = 50
    num_cates = 10
    max_size = 8
    data = generate_data(num_users, num_tuple, num_items, num_cates, max_size)
    generator = torchutils.fashion.getGenerator(mode="Identity")
    assert (generator(data) == data).all()


def test_replace():
    num_users = 10
    num_tuple = 300
    num_items = 50
    num_cates = 10
    max_size = 8
    ratio = 10
    data = generate_data(num_users, num_tuple, num_items, num_cates, max_size)
    generator = torchutils.fashion.getGenerator(mode="RandomReplace", num_replace=1, ratio=10)
    assert (generator(data) != data.repeat(ratio, axis=0)).sum() == (ratio * len(data))

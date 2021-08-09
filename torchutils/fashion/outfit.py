from . import utils


class OutfitTuple(object):
    def __init__(self, data):
        super().__init__()
        self.data = data
        item_list, all_items = utils.get_item_list(data)
        self.item_list = item_list
        self.all_items = all_items
        self.max_size = utils.infer_max_size(data)
        self.num_type = utils.infer_num_type(data)
        self.num_user = len(set(data[:, 0]))

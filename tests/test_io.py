import os
import shutil
import tempfile
import unittest

import torch
from torch import nn

from torchutils.misc import load_pretrained


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


class TestLoadPretrained(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.temp_dir, "checkpoint.pth")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_load_pretrained_from_path(self):
        # Save a checkpoint
        torch.save(self.model.state_dict(), self.checkpoint_path)

        # Load the checkpoint
        loaded_model = load_pretrained(SimpleModel(), self.checkpoint_path)

        # Check if the state_dicts are equal
        for param1, param2 in zip(self.model.state_dict().values(), loaded_model.state_dict().values()):
            self.assertTrue(torch.equal(param1, param2))

    def test_load_pretrained_from_state_dict(self):
        # Save a state_dict
        state_dict = self.model.state_dict()

        # Load the state_dict
        loaded_model = load_pretrained(SimpleModel(), state_dict)

        # Check if the state_dicts are equal
        for param1, param2 in zip(state_dict.values(), loaded_model.state_dict().values()):
            self.assertTrue(torch.equal(param1, param2))

    def test_load_pretrained_with_unmatched_keys(self):
        # Modify the model to create unmatched keys
        modified_model = SimpleModel()
        modified_model.fc = nn.Linear(10, 2)

        # Save a checkpoint
        torch.save(modified_model.state_dict(), self.checkpoint_path)

        # Load the checkpoint
        loaded_model = load_pretrained(SimpleModel(), self.checkpoint_path)

        # Check if the state_dicts are not equal due to unmatched keys
        for param1, param2 in zip(self.model.state_dict().values(), loaded_model.state_dict().values()):
            self.assertFalse(torch.equal(param1, param2))


if __name__ == "__main__":
    unittest.main()

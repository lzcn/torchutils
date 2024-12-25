import os
import shutil
import tempfile
import unittest
import warnings

import torch
from torch import nn

from torchutils.io import ModelSaver


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


class TestModelSaver(unittest.TestCase):
    def setUp(self):
        self.dirname = tempfile.mkdtemp()
        self.model = SimpleModel()
        self.model_saver = ModelSaver(
            dirname=self.dirname, filename_prefix="test_model", n_saved=3, save_latest=True, save_best=True
        )

    def tearDown(self):
        shutil.rmtree(self.dirname)

    def test_save_model(self):
        # Save model with different scores
        for epoch in range(5):
            score = epoch * 0.1
            self.model_saver.save(self.model, score, epoch)

        # Check if the correct number of files are saved
        saved_files = os.listdir(self.dirname)
        self.assertEqual(len(saved_files), 5)  # 3 best + 1 latest + 1 best

        # Check if the latest model is saved correctly
        latest_model_path = os.path.join(self.dirname, "test_model_latest.pt")
        self.assertTrue(os.path.exists(latest_model_path))

        # Check if the best model is saved correctly
        best_model_path = os.path.join(self.dirname, "test_model_best.pt")
        self.assertTrue(os.path.exists(best_model_path))

    def test_save_best_model(self):
        # Save model with different scores
        scores = [0.1, 0.4, 0.2, 0.5, 0.3]
        for epoch, score in enumerate(scores):
            self.model_saver.save(self.model, score, epoch)

        # Check if the best model is saved correctly
        best_model_path = os.path.join(self.dirname, "test_model_best.pt")
        self.assertTrue(os.path.exists(best_model_path))

        # Load the best model and check its state_dict
        best_model = SimpleModel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            best_model.load_state_dict(torch.load(best_model_path))
        self.assertEqual(self.model.state_dict().keys(), best_model.state_dict().keys())


if __name__ == "__main__":
    unittest.main()

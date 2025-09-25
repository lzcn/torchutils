Examples
========

Practical examples of using the available torchutils modules.

Basic Usage
-----------

Device Management
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from torchutils.ops import to

   # Move nested data structures to GPU
   data = {
       'input': torch.randn(32, 3, 224, 224),
       'target': torch.randint(0, 10, (32,)),
       'metadata': ['sample1', 'sample2']  # strings are preserved
   }
   
   data_gpu = to(data, 'cuda')

Model Backbones
~~~~~~~~~~~~~~~

.. code-block:: python

   from torchutils.backbones import backbone

   # Load a ResNet50 with ImageNet weights
   model, output_dim = backbone('resnet50', weights='IMAGENET1K_V1')
   print(f"Output dimension: {output_dim}")

   # Use in your model
   import torch.nn as nn
   
   class Classifier(nn.Module):
       def __init__(self, num_classes=10):
           super().__init__()
           self.backbone, backbone_dim = backbone('resnet50')
           self.classifier = nn.Linear(backbone_dim, num_classes)
       
       def forward(self, x):
           features = self.backbone(x)
           return self.classifier(features)

Pretrained Loading
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from torchutils import load_pretrained

   model = Classifier()

   # Load from a checkpoint file
   load_pretrained(model, 'artifacts/best_model.ckpt')

   # Or directly from a state dict (e.g. WandB artifact)
   state_dict = torch.load('snapshot.pt', map_location='cpu')
   load_pretrained(model, state_dict)

`load_pretrained` filters out unexpected or shape-mismatched keys, logs
missing entries for review, and supports a ``strict=True`` flag to enforce
perfect matches during upgrades.

Logging
~~~~~~~

.. code-block:: python

   from torchutils.logger import config, get_logger

   # Configure the root logger once (optional, but useful for global settings)
   config(
       level="INFO",               # default level applied to handlers
       stream_formatter="concise",  # pick one of the built-in formats
       log_file="train.log",        # set to None to disable file logging
   )

   # Create a module-specific logger that respects the global config
   logger = get_logger("training")

   logger.info("Starting training...")
   logger.warning("Low learning rate detected")
   logger.error("Training failed!")

`config` controls the shared logging configuration (handlers, formats, levels),
while `get_logger` gives you a lightweight logger instance tied to the calling
module. Call `config` once during application start-up, then use `get_logger`
wherever you need to emit messages.

Model Saving
~~~~~~~~~~~~

.. code-block:: python

   from torchutils.io import ModelSaver

   # Initialize model saver
   saver = ModelSaver('/path/to/checkpoints', keep_num=5)
   
   # Save model checkpoint
   saver.save({
       'model': model.state_dict(),
       'optimizer': optimizer.state_dict(),
       'epoch': epoch,
       'loss': loss.item()
   }, epoch)

Advanced Usage
--------------

Distributed Training
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torchutils.distributed import rank_zero_only

   @rank_zero_only
   def log_metrics(metrics):
       # This function only runs on the main process
       print(f"Metrics: {metrics}")

   # Use in training loop
   for epoch in range(num_epochs):
       # ... training code ...
       log_metrics({'loss': avg_loss, 'acc': accuracy})

# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from src.utils.visualization import fig_to_numpy
from .base_model import BaseModule


class ModelNet40Base(BaseModule):
    """Base model class."""

    def data_dict_to_input(self, data_dict, **kwargs) -> Any:
        """Convert data dictionary to appropriate input for the model."""
        points = data_dict["vertices"].to(self.device)
        label = data_dict["class"].to(self.device)
        return points, label

    def loss_dict(self, data_dict, **kwargs) -> Dict:
        """Compute the loss dictionary for the model."""
        points, labels = self.data_dict_to_input(data_dict)
        pred = self(points)
        if "loss_fn" in kwargs:
            loss_fn = kwargs["loss_fn"]
        else:
            loss_fn = nn.CrossEntropyLoss()
        return_dict = {}
        loss = loss_fn(pred, labels)
        return_dict["loss"] = loss
        return return_dict

    @torch.no_grad()
    def eval_dict(self, data_dict, **kwargs) -> Dict:
        """Compute the evaluation dictionary for the model."""
        points, labels = self.data_dict_to_input(data_dict)
        pred = self(points)
        if "loss_fn" in kwargs:
            loss_fn = kwargs["loss_fn"]
        else:
            loss_fn = nn.CrossEntropyLoss()
        return_dict = {}
        loss = loss_fn(pred, labels)
        return_dict["loss"] = loss
        # Add pred_label and target_label to the return dictionary
        return_dict["_pred_label"] = pred.argmax(dim=1).cpu().numpy()
        return_dict["_target_label"] = labels.cpu().numpy()
        return return_dict
    
    def post_eval_epoch(self, eval_dict: dict, datamodule, **kwargs) -> Tuple[Dict, Dict, Dict]:
        """Post evaluation epoch hook."""
        # Compute the confusion matrix
        from sklearn.metrics import confusion_matrix
        pred_label = eval_dict["_pred_label"]
        target_label = eval_dict["_target_label"]
        confusion_matrix = confusion_matrix(target_label, pred_label)
        eval_dict["confusion_matrix"] = confusion_matrix
        accuracy = confusion_matrix.trace() / confusion_matrix.sum()
        eval_dict["total_accuracy"] = accuracy

        # Plot confusion matrix
        fig = plt.figure(figsize=(10, 8))
        # matplotlib heatmap
        plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        im = fig_to_numpy(fig)
    
        return eval_dict, {"confusion_matrix": im}, {}

    def image_pointcloud_dict(self, data_dict, datamodule) -> Tuple[Dict, Dict]:
        """Compute the image dict and pointcloud dict for the model."""
        return_dict = {}
        image_dict = {}
        return image_dict, return_dict

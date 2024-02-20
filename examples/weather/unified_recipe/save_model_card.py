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

import os
import json

def save_model_card(
    model,
    predicted_variables,
    unpredicted_variables,
    save_path,
    readme=None,
    ):
    """
    Save model card for model global weather model.

    The model card will be a directory with the following files:
    - model.mdlus: The model file.
    - variables.json: A json file with the predicted and unpredicted variables.
    - README.md: A readme file with the readme text.

    Parameters
    ----------
    model : modulus.Module
        Model to save model card for.
    predicted_variables : list
        List of predicted variables.
    unpredicted_variables : list
        List of unpredicted variables.
    readme : str
        readme text for model card.
    """

    # Create model card directory
    os.makedirs(save_path, exist_ok=True)

    # Save model mdlus file
    model.save(
        os.path.join(save_path, "model.mdlus"),
    )

    # Save json files with predicted and unpredicted variables to a json file
    # Make variables json serializable
    formatted_predicted_variables = []
    for variable in predicted_variables:
        if isinstance(variable, str):
            formatted_predicted_variables.append(variable)
        else:
            formatted_predicted_variables.append((variable[0], tuple(variable[1])))
    formatted_unpredicted_variables = []
    for variable in unpredicted_variables:
        if isinstance(variable, str):
            formatted_unpredicted_variables.append(variable)
        else:
            formatted_unpredicted_variables.append((variable[0], tuple(variable[1])))
    json_dict = {
        "predicted_variables": formatted_predicted_variables,
        "unpredicted_variables": formatted_unpredicted_variables,
    }
    with open(os.path.join(save_path, "variables.json"), "w") as f:
        json.dump(json_dict, f)

    # Save readme
    if readme is not None:
        with open(os.path.join(save_path, "README.md"), "w") as f:
            f.write(readme)

if __name__ == "__main__":
    main()

# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from treelib import Tree
from typing import Dict, List, Optional, Union


class ProcessGroupNode:
    """
    Class to store the attributes of a distributed process group

    Attributes
    ----------
    name : str
        Name of the process group
    size : Optional[int]
        Optional, number of processes in the process group
    orthogonal_group : Optional[str]
        Optional, name of an orthogonal process group to create
    """

    def __init__(
        self,
        name: str,
        size: Optional[int] = None,
        orthogonal_group: Optional[str] = None,
    ):
        """
        Constructor for the ProcessGroupNode class

        Parameters
        ----------
        name : str
            Name of the process group
        size : Optional[int]
            Optional, size of the process group
        orthogonal_group : Optional[str]
            Optional, name of an orthogonal process group to create
        """
        self.name = name
        self.size = size
        self.orthogonal_group = orthogonal_group

    def __str__(self):
        """
        String representation of the process group node

        Returns
        -------
        str
            String representation of the process group node
        """
        return (
            "ProcessGroupNode("
            f"name={self.name}, "
            f"size={self.size}, "
            f"orthogonal_group={self.orthogonal_group})"
        )

    def __repr__(self):
        """
        String representation of the process group node

        Returns
        -------
        str
            String representation of the process group node
        """
        return self.__str__()


class ProcessGroupConfig:
    """
    Class to define the configuration of a model's parallel process group structure as a
    tree. Each node of the tree is of type `ProcessGroupNode`.

    Once the process group config structure (i.e, the tree structure) is set, it is
    sufficient to set only the sizes for each leaf process group. Then, the size of
    every parent group can be automatically computed as the product reduction of the
    sub-tree of that parent group node.

    Examples
    --------
    >>> from modulus.distributed.config import ProcessGroupNode, ProcessGroupConfig
    >>>
    >>> # Create model parallel group with data parallel as the orthogonal group
    >>> mp = ProcessGroupNode("world")
    >>> mp = ProcessGroupNode("model_parallel", orthogonal_group="data_parallel")
    >>>
    >>> # Create spatial and channel parallel sub-groups
    >>> config.add_node(ProcessGroupNode("model_parallel"), parent=mp)
    >>> config.add_node(ProcessGroupNode("data_parallel"), parent=mp)
    >>>
    >>> # Create the process group config with the highest level process group
    >>> config = ProcessGroupConfig(mp)
    >>>
    >>> # Create spatial and channel parallel sub-groups
    >>> config.add_node(ProcessGroupNode("spatial_parallel"), parent=mp)
    >>> config.add_node(ProcessGroupNode("channel_parallel"), parent="model_parallel")
    >>>
    >>> pg_config.leaf_groups()
    ['spatial_parallel', 'channel_parallel']
    >>>
    >>> # Set leaf group sizes
    >>> group_sizes = {"channel_parallel": 3, "spatial_parallel": 2}
    >>> pg_config.set_leaf_group_sizes(group_sizes)  # Update all parent group sizes too
    >>> pg_config.get_node("model_parallel").size
    6
    """

    def __init__(self, node: ProcessGroupNode):
        """
        Constructor to the ProcessGroupConfig class

        Parameters
        ----------
        node : ProcessGroupNode
            Root node of the tree, typically would be 'model_parallel'
            Note, it is generally recommended to always set the orthogonal_group for
            the 'model_parallel' group to be 'data_parallel' to aid with distributed
            data parallel training
        """
        self.root = node
        self.root_id = node.name
        self.tree = Tree()
        self.tree.create_node(node.name, node.name, data=node)

    def add_node(self, node: ProcessGroupNode, parent=Union[str, ProcessGroupNode]):
        """
        Add a node to the process group config

        Parameters
        ----------
        node : ProcessGroupNode
            The new node to be added to the config
        parent : Union[str, ProcessGroupNode]
            Parent node of the node to be added. Should already be in the config.
            If str, it is the name of the parent node. Otherwise, the parent
            ProcessGroupNode itself.
        """
        if isinstance(parent, ProcessGroupNode):
            parent = parent.name
        self.tree.create_node(node.name, node.name, data=node, parent=parent)

    def get_node(self, name: str) -> ProcessGroupNode:
        """
        Method to get the node given the name of the node

        Parameters
        ----------
        name : str
            Name of the node to retrieve

        Returns
        -------
        ProcessGroupNode
            Node with the given name from the config
        """
        return self.tree.get_node(name).data

    def update_parent_sizes(self, verbose: bool = False) -> int:
        """
        Method to update parent node sizes after setting the sizes for each leaf node

        Parameters
        ----------
        verbose : bool
            If True, print a message each time a parent node size was updated

        Returns
        -------
        int
            Size of the root node
        """
        return _tree_product_reduction(self.tree, self.root_id, verbose=verbose)

    def leaf_groups(self) -> List[str]:
        """
        Get a list of all leaf group names

        Returns
        -------
        List[str]
            List of all leaf node names
        """
        return [n.identifier for n in self.tree.leaves()]

    def set_leaf_group_sizes(
        self, group_sizes: Dict[str, int], update_parent_sizes: bool = True
    ):
        """
        Set process group sizes for all leaf groups

        Parameters
        ----------
        group_sizes : Dict[str, int]
            Dictionary with a mapping of each leaf group name to its size
        update_parent_sizes : bool
            Update all parent group sizes based on the leaf group if True
            If False, only set the leaf group sizes.
        """
        for id, size in group_sizes.items():
            assert self.tree.contains(
                id
            ), f"Process group {id} is not in this process group config"
            node = self.tree.get_node(id)
            assert node.is_leaf(), f"Process group {id} is not a leaf group"
            node.data.size = size

        if update_parent_sizes:
            self.update_parent_sizes()


def _tree_product_reduction(tree, node_id, verbose=False):
    """
    Function to traverse a tree and compute the product reduction of
    the sub-tree for each node starting from `node_id`
    """
    children = tree.children(node_id)
    node = tree.get_node(node_id)
    if not children:
        assert node.data.size is not None, "Leaf nodes should have a valid size set"
        return node.data.size

    product = 1

    for child in children:
        product *= _tree_product_reduction(tree, child.identifier)

    if node.data.size != product:
        if verbose:
            print(
                "Updating size of node "
                f"{node.data.name} from {node.data.size} to {product}"
            )
        node.data.size = product

    return product

# src/darcy_automl_active_learning/model_selection/selection_strategies.py

from abc import ABC, abstractmethod
from typing import List, Tuple, Any

class BaseSelectionStrategy(ABC):
    """
    Base class for different model selection strategies.
    """

    @abstractmethod
    def select_candidates(self, data_desc: dict, model_registry: Any) -> List[Tuple[str, str]]:
        """
        Return a list of (model_name, candidate_key) pairs based on data_desc.

        Args:
            data_desc (dict): The dataset descriptor loaded from JSON
            model_registry (ModelRegistry or a generic type): 
                Provides access to model descriptors

        Returns:
            List[Tuple[str,str]]: e.g. [("FNO","candidate0"), ("AFNO","candidate1")]
        """
        pass


class SimpleSelectionStrategy(BaseSelectionStrategy):
    """
    An example strategy that always picks FNO, or picks a small set of models
    regardless of data. Extend as needed for real logic.
    """

    def select_candidates(self, data_desc: dict, model_registry: Any) -> List[Tuple[str, str]]:
        # A trivial approach: always select "FNO" as candidate0
        # Note: you could parse data_desc["data_structure"] to check dimension, etc.

        # Possibly do: if dimension=2 and geometry=grid => pick FNO, else pick DiffusionNet
        # For now, a simple example:
        selected = [("FNOWithDropout", "candidate0")]
        return selected

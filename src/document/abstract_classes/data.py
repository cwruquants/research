import numpy as np
import abc as ABC, abstract_classes

class Data(ABC):
    @abstractmethod
    def load_data(self, source: str) -> str:
        """Loads data from a given path."""
        
        pass

    @abstractmethod
    def clean_data(selc, data: str) -> str:
        """Cleans the raw data."""
        pass

    @abstractmethod
    def transform_DocumentAttr(self, str) -> 'DocumentAttr':
        """Transforms raw string data into a DocumentAttr object."""
        pass

from ...abstract_classes.attribute import DocumentAttr

class MatchingAgent:
    def __init__(self, keyword_doc: DocumentAttr, threshold = 0.7, csv_path = None):
        """
        Initialize a MatchingAgent that analyzes document exposure based on keywords.

        Args:
            keyword_doc (DocumentAttr): Document containing the keywords to match against
            threshold (float): Similarity threshold for matching (default: 0.7)
            csv_path (str, optional): Path to CSV file containing exposure words
        """
        self.keyword_doc = keyword_doc
        self.threshold = threshold
        self.exposure_list = []

        if csv_path:
            self.load_exposure_words(csv_path)

    def load_exposure_words(self, csv_path: str) -> None:
        """
        Load and process exposure words from a CSV file.

        Args:
            csv_path (str): Path to the CSV file containing exposure words
        """
        # TODO:
        # 1. Read CSV file
        # 2. Process words (cleanup, normalize)
        # 3. Store in self.exposure_list
        pass

    def cos_similarity(self, threshold: float = None, return_instances: bool = False, return_context: bool = False):
        """
        Calculate cosine similarity between documents and find matching instances.

        Args
            threshold (float, optional): Override default similarity threshold
            return_instances (bool): If True, return detailed match instances
            return_context (bool): If True, return context around matches

        Returns:
            ExposureResults: Object containing match statistics and instances
        """
        # TODO:
        # 1. Get effective threshold (passed or self.threshold)
        # 2. Calculate embeddings for both documents
        # 3. Compute cosine similarity
        # 4. Find matches above threshold
        # 5. If return_instances, collect context
        # 6. Return ExposureResults object
        pass

    def direct_match(self, return_instances: bool = False, return_context: bool = False):
        """
        Find exact matches between keywords and document text.

        Args:
            return_context (bool):
            return_instances (bool): If return_context is true, return match instances

        Returns:
            Number of matches and optionally match details
        """
        # TODO:
        # 1. Extract keywords from keyword_doc
        # 2. Find exact matches in target document
        # 3. Count matches
        # 4. If return_instances, collect match context
        # 5. Return results
        pass

    # def get_context_window(self, doc: DocumentAttr, match_position: int, window_size: int = 2) -> str:
    #     """
    #     Get context window around a match in the document.

    #     Returns:
    #         str: Context window around the match
    #     """
    #     # TODO:
    #     # 1. Find sentence containing match
    #     # 2. Get surrounding sentences based on window_size
    #     # 3. Join and return context
    #     pass

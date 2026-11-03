"""Cosine-similarity matching with all-MiniLM-L6-v2 (needs the [ml] extra).

MatchingAgent mixes this in; torch and sentence-transformers are imported only when a
cosine match runs.
"""

import time
from typing import Any, Dict, List, Tuple

from research.matching.results import ExposureResults, MatchInstance
from research.transcripts.tokenize import (
    document_to_bigram,
    document_to_sentence,
    document_to_word,
    is_bigram,
)


class CosineMatchingMixin:
    def _get_model(self):
        # torch and sentence_transformers are imported here so direct-only runs never load them
        if self.model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            # Device selection is silent to avoid disrupting progress bars
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.model.to(self.device)
        return self.model

    def _prepare_corpus(self, match_type: str) -> Tuple[List[str], int]:
        """Prepares the corpus for matching based on the match type."""
        document_words = document_to_word(self.document)
        if match_type == "word":
            return document_words, len(document_words)

        document_bigrams = document_to_bigram(self.document)
        if match_type == "bigram":
            return document_bigrams, 0

        if match_type == "hybrid":
            return document_words + document_bigrams, len(document_words)

        raise ValueError(f"Unsupported match_type: {match_type}. Use 'word', 'bigram', or 'hybrid'.")

    def _process_hits(
        self,
        hits: List[Dict[str, Any]],
        keyword: str,
        corpus: List[str],
        sentences: List[str],
        num_words: int,
        threshold: float,
    ) -> Tuple[List[MatchInstance], List[MatchInstance]]:
        """Processes search hits to create direct and cosine match instances."""
        direct_matches = []
        cosine_matches = []

        document_words = document_to_word(self.document) if num_words > 0 else []

        for hit in hits[0]:
            if hit["score"] < threshold:
                continue

            corpus_id = int(hit["corpus_id"])
            score = float(hit["score"])

            if num_words > 0 and corpus_id < num_words:
                # This is a word match
                position = corpus_id
                matched_text = document_words[position]
            else:
                # This is a bigram match
                position = corpus_id - num_words
                matched_text = corpus[corpus_id]

            context = self._get_context(position, sentences)

            match_instance = MatchInstance(
                keyword=keyword,
                matched_text=matched_text,
                context=context,
                position=position,
                similarity_score=score,
            )

            if score > 0.99:
                match_instance.similarity_score = 1.0
                direct_matches.append(match_instance)
            else:
                cosine_matches.append(match_instance)

        return direct_matches, cosine_matches

    def _cos_similarity(
        self,
        match_type: str = "word",
        threshold: float | None = None,
        exclude_duplicates: bool = True,
        show_progress: bool = False,
    ) -> ExposureResults:
        """
        Calculate cosine similarity between keywords and document text.
        Args:
            match_type (str): Type of matching ("word", "bigram", or "hybrid")
            threshold (float, optional): Override default similarity threshold
            exclude_duplicates (bool): Exclude duplicate matches (matches with same position)
        Returns:
            ExposureResults: Object containing cosine similarity match results
        """
        if not self.document or not self.document.text:
            raise ValueError("No document provided for matching")

        from sentence_transformers import util

        start_time = time.time()
        model = self._get_model()

        final_threshold = threshold if threshold is not None else self.cos_threshold

        sentences = document_to_sentence(self.document)
        total_sentences = len(sentences)

        results = ExposureResults(
            keyword_doc=self.keywords_list,
            keyword_doc_name=self.keyword_doc_name,
            earnings_call=self.document,
            earnings_call_name=self.document_name,
            total_sentences_in_document=total_sentences,
            cosine_threshold=final_threshold,
        )

        corpus, num_words = self._prepare_corpus(match_type)

        if not corpus:
            return results

        corpus_lower = [item.lower() for item in corpus]
        corpus_embeddings = model.encode(corpus_lower, convert_to_tensor=True, device=self.device)

        for keyword in self.keywords_list:
            try:
                query_embedding = model.encode(keyword.lower(), convert_to_tensor=True, device=self.device)
                hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=len(corpus))

                direct_matches, cosine_matches = self._process_hits(
                    hits, keyword, corpus, sentences, num_words, final_threshold
                )

                if direct_matches:
                    results.add_direct_matches(keyword, direct_matches)
                if cosine_matches:
                    results.add_cosine_matches(keyword, cosine_matches)

            except Exception as e:
                print(f"Warning: Error processing keyword '{keyword}': {str(e)}")

        if exclude_duplicates:
            # The logic for excluding duplicates remains the same
            word_matches = {}
            bigram_matches = {}

            for km in results.keyword_matches.values():
                for match in km.direct_matches + km.cosine_matches:
                    if match.position is None:
                        continue

                    score = match.similarity_score if match.similarity_score is not None else 1.0

                    if is_bigram(match.matched_text):
                        if (
                            match.position not in bigram_matches
                            or score > bigram_matches[match.position].similarity_score
                        ):
                            bigram_matches[match.position] = match
                    else:
                        if (
                            match.position not in word_matches
                            or score > word_matches[match.position].similarity_score
                        ):
                            word_matches[match.position] = match

            filtered_bigram_matches = {
                bigram_pos: bigram_match
                for bigram_pos, bigram_match in bigram_matches.items()
                if not any(abs(bigram_pos - word_pos) <= 1 for word_pos in word_matches)
            }

            final_matches = {**word_matches, **filtered_bigram_matches}

            filtered_results = ExposureResults(
                keyword_doc=self.keywords_list,
                keyword_doc_name=self.keyword_doc_name,
                earnings_call=self.document,
                earnings_call_name=self.document_name,
                total_sentences_in_document=total_sentences,
                cosine_threshold=final_threshold,
            )

            for match in final_matches.values():
                if match.similarity_score and match.similarity_score > 0.99:
                    filtered_results.add_direct_matches(match.keyword, [match])
                else:
                    filtered_results.add_cosine_matches(match.keyword, [match])

            filtered_results.runtime_seconds = time.time() - start_time
            return filtered_results

        results.runtime_seconds = time.time() - start_time
        return results

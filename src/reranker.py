from sentence_transformers import CrossEncoder


class DocumentReranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        CrossEncoder-based document reranker.

        The CrossEncoder receives:
            (query, document)

        and returns a relevance score for each pair.
        """

        self.model = CrossEncoder(
            model_name
        )


    # =====================================================
    # GET RELEVANCE SCORES
    # =====================================================

    def get_scores(
        self,
        query,
        documents
    ):
        """
        Calculate relevance scores for all documents.

        Returns:
            list[float]
        """

        if not documents:
            return []

        pairs = [
            (
                query,
                document.page_content
            )
            for document in documents
        ]

        scores = self.model.predict(
            pairs
        )

        return [
            float(score)
            for score in scores
        ]


    # =====================================================
    # RERANK DOCUMENTS
    # =====================================================

    def rerank(
        self,
        query,
        documents,
        top_k=4
    ):
        """
        Rerank documents based on CrossEncoder scores.

        Returns:
            Top-k documents ordered by relevance.
        """

        if not documents:
            return []

        scores = self.get_scores(
            query,
            documents
        )

        ranked = sorted(
            zip(
                scores,
                documents
            ),
            key=lambda x: x[0],
            reverse=True
        )

        return [
            document
            for score, document
            in ranked[:top_k]
        ]

from collections import defaultdict


class RRFFusionRetriever:
    """
    Reciprocal Rank Fusion (RRF)

    Combines results from:
    1. FAISS semantic search
    2. BM25 keyword search
    """

    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
        k=60
    ):

        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

        # RRF constant
        self.k = k

    def invoke(
        self,
        query,
        top_k=8
    ):

        # =========================================
        # FAISS results
        # =========================================

        vector_documents = (
            self.vector_retriever.invoke(
                query
            )
        )

        # =========================================
        # BM25 results
        # =========================================

        bm25_documents = (
            self.bm25_retriever.invoke(
                query
            )
        )

        # =========================================
        # RRF scoring
        # =========================================

        scores = defaultdict(float)

        documents = {}

        # -----------------------------------------
        # FAISS ranking
        # -----------------------------------------

        for rank, document in enumerate(
            vector_documents,
            start=1
        ):

            document_id = self._document_id(
                document
            )

            scores[document_id] += (
                1 / (self.k + rank)
            )

            documents[
                document_id
            ] = document

        # -----------------------------------------
        # BM25 ranking
        # -----------------------------------------

        for rank, document in enumerate(
            bm25_documents,
            start=1
        ):

            document_id = self._document_id(
                document
            )

            scores[document_id] += (
                1 / (self.k + rank)
            )

            documents[
                document_id
            ] = document

        # =========================================
        # Sort by RRF score
        # =========================================

        ranked_documents = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        # =========================================
        # Return top documents
        # =========================================

        results = []

        for document_id, score in ranked_documents[
            :top_k
        ]:

            results.append(
                documents[document_id]
            )

        return results

    @staticmethod
    def _document_id(
        document
    ):

        """
        Creates a unique identifier
        for a document chunk.
        """

        source = document.metadata.get(
            "source",
            ""
        )

        page = document.metadata.get(
            "page",
            ""
        )

        content = document.page_content

        return (
            f"{source}|"
            f"{page}|"
            f"{hash(content)}"
        )
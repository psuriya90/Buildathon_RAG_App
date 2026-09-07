
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .reranker import DocumentReranker
from .relevance_router import RelevanceRouter
from .web_search import web_search


class RAGPipeline:

    def __init__(
        self,
        fusion_retriever
    ):

        self.fusion_retriever = (
            fusion_retriever
        )

        self.reranker = (
            DocumentReranker()
        )

        self.router = (
            RelevanceRouter(
                threshold=0.20
            )
        )

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )


    # =====================================================
    # ASK
    # =====================================================

    def ask(
        self,
        question
    ):

        # -----------------------------------------------
        # STEP 1: RETRIEVE DOCUMENTS
        # -----------------------------------------------

        documents = (
            self.fusion_retriever.invoke(
                question
            )
        )

        if not documents:

            # If no documents were retrieved,
            # directly use web search.

            answer = (
                self.answer_from_web(
                    question
                )
            )

            return {
                "answer": answer,
                "source": "Web Search",
                "score": None
            }


        # -----------------------------------------------
        # STEP 2: CALCULATE RERANKING SCORES
        # -----------------------------------------------

        scores = (
            self.reranker.get_scores(
                question,
                documents
            )
        )

        # -----------------------------------------------
        # STEP 3: GET BEST SCORE
        # -----------------------------------------------

        best_score = (
            self.router.get_best_score(
                scores
            )
        )


        # -----------------------------------------------
        # STEP 4: RELEVANCE ROUTING
        # -----------------------------------------------

        if not self.router.is_relevant(
            scores
        ):

            # Document is not sufficiently relevant.
            # Use Tavily / web search.

            answer = (
                self.answer_from_web(
                    question
                )
            )

            return {
                "answer": answer,
                "source": "Web Search",
                "score": best_score
            }


        # -----------------------------------------------
        # STEP 5: RERANK DOCUMENTS
        # -----------------------------------------------

        relevant_documents = (
            self.reranker.rerank(
                question,
                documents,
                top_k=4
            )
        )

        if not relevant_documents:

            answer = (
                self.answer_from_web(
                    question
                )
            )

            return {
                "answer": answer,
                "source": "Web Search",
                "score": best_score
            }


        # -----------------------------------------------
        # STEP 6: BUILD CONTEXT
        # -----------------------------------------------

        context = "\n\n".join(
            document.page_content
            for document
            in relevant_documents
        )


        # -----------------------------------------------
        # STEP 7: GENERATE ANSWER
        # -----------------------------------------------

        answer = (
            self.answer_from_documents(
                question,
                context
            )
        )


        return {
            "answer": answer,
            "source": "RAG",
            "score": best_score
        }


    # =====================================================
    # ANSWER FROM DOCUMENTS
    # =====================================================

    def answer_from_documents(
        self,
        question,
        context
    ):

        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful RAG assistant.

Answer the user's question using ONLY
the supplied document context.

If the answer is not available in the
context, clearly say that the information
is not available in the uploaded document.

Do not invent information.

Document Context:
{context}

Question:
{question}

Answer:
"""
        )

        chain = (
            prompt
            | self.llm
        )

        response = chain.invoke(
            {
                "context": context,
                "question": question
            }
        )

        return response.content


    # =====================================================
    # ANSWER FROM WEB
    # =====================================================

    def answer_from_web(
        self,
        question
    ):

        try:

            search_results = (
                web_search(
                    question
                )
            )

        except Exception as e:

            return (
                "I could not perform web search "
                f"for this question. Error: {str(e)}"
            )


        # -----------------------------------------------
        # EXTRACT WEB RESULTS
        # -----------------------------------------------

        web_context = "\n\n".join(
            result.get(
                "content",
                ""
            )
            for result in search_results.get(
                "results",
                []
            )
            if result.get(
                "content"
            )
        )


        if not web_context:

            return (
                "I could not find reliable "
                "information for this question "
                "through web search."
            )


        # -----------------------------------------------
        # WEB PROMPT
        # -----------------------------------------------

        prompt = ChatPromptTemplate.from_template(
            """
You are a web research assistant.

Answer the question using the supplied
web search results.

Use only information supported by the
search results.

Do not invent information.

Web Search Results:
{context}

Question:
{question}

Answer:
"""
        )


        chain = (
            prompt
            | self.llm
        )


        response = chain.invoke(
            {
                "context": web_context,
                "question": question
            }
        )


        return response.content


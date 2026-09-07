import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.document_loader import load_document
from src.text_splitter import split_documents

from src.vector_store import (
    create_vector_store,
    save_vector_store
)

from src.bm25_retriever import (
    create_bm25_retriever
)

from src.fusion_retriever import (
    RRFFusionRetriever
)

from src.rag_pipeline import (
    RAGPipeline
)

from src.excel_logger import (
    log_question
)


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# DIRECTORIES
# =========================================================

UPLOAD_DIR = Path("data/uploads")
VECTORSTORE_DIR = Path("data/vectorstore")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VECTORSTORE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

if "document_ready" not in st.session_state:
    st.session_state.document_ready = False

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


# =========================================================
# TITLE
# =========================================================

st.title("🤖 RAG Document Assistant")

st.write(
    "Upload a PDF, TXT or DOCX document "
    "and ask questions using Hybrid RAG."
)


# =========================================================
# CHECK OPENAI API KEY
# =========================================================

openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:

    st.error(
        "❌ OPENAI_API_KEY is not configured."
    )

    st.info(
        "Please add OPENAI_API_KEY to your .env file."
    )

    st.stop()


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload your document",
    type=[
        "pdf",
        "txt",
        "docx"
    ]
)


# =========================================================
# PROCESS DOCUMENT
# =========================================================

if uploaded_file is not None:

    st.success(
        f"✅ Successfully uploaded: {uploaded_file.name}"
    )

    st.write(
        f"**File size:** {uploaded_file.size} bytes"
    )

    # -----------------------------------------------------
    # Save uploaded file
    # -----------------------------------------------------

    file_path = UPLOAD_DIR / uploaded_file.name

    try:

        with open(
            file_path,
            "wb"
        ) as file:

            file.write(
                uploaded_file.getvalue()
            )

        st.info(
            f"📁 File saved to: `{file_path}`"
        )

    except Exception as e:

        st.error(
            "❌ Unable to save uploaded file."
        )

        st.exception(e)

        st.stop()


    # -----------------------------------------------------
    # Process button
    # -----------------------------------------------------

    process_button = st.button(
        "🚀 Process Document",
        type="primary"
    )


    if process_button:

        # Reset previous state
        st.session_state.document_ready = False
        st.session_state.rag_pipeline = None
        st.session_state.uploaded_file_name = None


        # =================================================
        # PROCESS DOCUMENT
        # =================================================

        try:

            with st.status(
                "📄 Processing document...",
                expanded=True
            ) as status:

                # =========================================
                # STEP 1 - LOAD DOCUMENT
                # =========================================

                st.write(
                    "Step 1/6: Loading document..."
                )

                documents = load_document(
                    str(file_path)
                )

                if not documents:

                    raise ValueError(
                        "No content could be extracted "
                        "from the uploaded document."
                    )

                st.write(
                    f"✅ Loaded {len(documents)} "
                    f"document sections/pages."
                )


                # =========================================
                # STEP 2 - SPLIT DOCUMENT
                # =========================================

                st.write(
                    "Step 2/6: Splitting document..."
                )

                chunks = split_documents(
                    documents
                )

                if not chunks:

                    raise ValueError(
                        "Document splitting returned "
                        "zero chunks."
                    )

                st.write(
                    f"✅ Created {len(chunks)} chunks."
                )


                # =========================================
                # STEP 3 - CREATE FAISS VECTOR STORE
                # =========================================

                st.write(
                    "Step 3/6: Creating FAISS "
                    "vector store..."
                )

                vector_store = create_vector_store(
                    chunks
                )

                save_vector_store(
                    vector_store,
                    str(VECTORSTORE_DIR)
                )

                st.write(
                    "✅ FAISS vector store created "
                    "and saved."
                )


                # =========================================
                # STEP 4 - FAISS RETRIEVER
                # =========================================

                st.write(
                    "Step 4/6: Creating FAISS retriever..."
                )

                vector_retriever = (
                    vector_store.as_retriever(
                        search_kwargs={
                            "k": 5
                        }
                    )
                )

                st.write(
                    "✅ FAISS retriever ready."
                )


                # =========================================
                # STEP 5 - BM25 RETRIEVER
                # =========================================

                st.write(
                    "Step 5/6: Creating BM25 retriever..."
                )

                bm25_retriever = (
                    create_bm25_retriever(
                        chunks
                    )
                )

                st.write(
                    "✅ BM25 retriever ready."
                )


                # =========================================
                # STEP 6 - RRF FUSION
                # =========================================

                st.write(
                    "Step 6/6: Creating "
                    "RRF Fusion Retriever..."
                )

                fusion_retriever = (
                    RRFFusionRetriever(
                        vector_retriever=vector_retriever,
                        bm25_retriever=bm25_retriever
                    )
                )

                st.write(
                    "✅ FAISS + BM25 fusion ready."
                )


                # =========================================
                # CREATE RAG PIPELINE
                # =========================================

                st.write(
                    "Creating RAG pipeline..."
                )

                rag_pipeline = RAGPipeline(
                    fusion_retriever
                )


                # =========================================
                # SAVE PIPELINE IN SESSION STATE
                # =========================================

                st.session_state.rag_pipeline = (
                    rag_pipeline
                )

                st.session_state.document_ready = (
                    True
                )

                st.session_state.uploaded_file_name = (
                    uploaded_file.name
                )


                # =========================================
                # PROCESSING COMPLETE
                # =========================================

                status.update(
                    label=(
                        "✅ Document processing "
                        "completed!"
                    ),
                    state="complete"
                )


            # Success message outside status
            st.success(
                "🎉 Your document is ready! "
                "You can now ask questions below."
            )

        except Exception as e:

            st.session_state.document_ready = False
            st.session_state.rag_pipeline = None

            st.error(
                "❌ Document processing failed."
            )

            st.error(
                f"Error: {str(e)}"
            )

            st.exception(e)


# =========================================================
# CHATBOT
# =========================================================

if (
    st.session_state.document_ready
    and st.session_state.rag_pipeline is not None
):

    st.divider()

    st.success(
        f"🤖 RAG chatbot is ready for "
        f"**{st.session_state.uploaded_file_name}**"
    )

    st.subheader(
        "💬 Ask Questions"
    )

    st.write(
        "Ask questions about the uploaded document."
    )


    # =====================================================
    # CHAT INPUT
    # =====================================================

    question = st.chat_input(
        "Ask a question..."
    )


    if question:

        # ================================================
        # USER MESSAGE
        # ================================================

        with st.chat_message("user"):

            st.write(question)


        # ================================================
        # ASSISTANT MESSAGE
        # ================================================

        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Searching document and generating answer..."
            ):

                try:

                    # ====================================
                    # ASK RAG PIPELINE
                    # ====================================

                    result = (
                        st.session_state
                        .rag_pipeline
                        .ask(question)
                    )


                    # ====================================
                    # HANDLE RESULT
                    # ====================================

                    if isinstance(result, dict):

                        answer = result.get(
                            "answer",
                            "No answer was generated."
                        )

                        source = result.get(
                            "source",
                            "Not available"
                        )

                        score = result.get(
                            "score",
                            None
                        )

                    else:

                        # If RAGPipeline.ask()
                        # returns a string
                        answer = str(result)

                        source = "Not available"

                        score = None


                    # ====================================
                    # DISPLAY ANSWER
                    # ====================================

                    st.write(answer)


                    # ====================================
                    # DISPLAY SOURCE
                    # ====================================

                    if source:

                        st.caption(
                            f"📚 Source: {source}"
                        )


                    # ====================================
                    # DISPLAY SCORE
                    # ====================================

                    if score is not None:

                        try:

                            st.caption(
                                f"🎯 Relevance score: "
                                f"{float(score):.4f}"
                            )

                        except (
                            ValueError,
                            TypeError
                        ):

                            st.caption(
                                f"🎯 Relevance score: "
                                f"{score}"
                            )


                    # ====================================
                    # SAVE QUESTION TO EXCEL
                    # ====================================

                    try:

                        log_question(
                            question=question,
                            answer=answer,
                            source=source
                        )

                        st.success(
                            "✅ Question and answer "
                            "saved to Excel."
                        )

                    except Exception as log_error:

                        st.warning(
                            "⚠️ Answer generated successfully, "
                            "but Excel logging failed."
                        )

                        st.caption(
                            f"Logging error: {log_error}"
                        )


                except Exception as e:

                    st.error(
                        "❌ Error while answering question."
                    )

                    st.exception(e)


# =========================================================
# INITIAL MESSAGE
# =========================================================

elif uploaded_file is None:

    st.info(
        "👆 Please upload a PDF, TXT or DOCX document "
        "to start."
    )

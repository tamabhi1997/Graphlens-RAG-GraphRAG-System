"""Streamlit entry point."""

import streamlit as st

from graphlens.pipelines.rag import answer_query


def main() -> None:
    st.set_page_config(page_title="GraphLens", layout="wide")
    st.title("GraphLens")
    st.caption("RAG playground for your vector-backed data project.")

    query = st.text_input("Ask a question")
    if st.button("Run") and query:
        response = answer_query(query)
        st.write(response)


if __name__ == "__main__":
    main()

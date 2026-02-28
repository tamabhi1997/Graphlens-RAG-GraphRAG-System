from graphlens.pipelines.rag import answer_query


def test_answer_query_returns_text() -> None:
    result = answer_query("test query")
    assert "test query" in result

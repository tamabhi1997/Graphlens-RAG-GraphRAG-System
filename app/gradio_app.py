"""Gradio entry point."""

import gradio as gr

from graphlens.pipelines.rag import answer_query


def run(query: str) -> str:
    return answer_query(query)


demo = gr.Interface(
    fn=run,
    inputs=gr.Textbox(label="Question"),
    outputs=gr.Textbox(label="Answer"),
    title="GraphLens",
    description="Minimal Gradio shell for a RAG workflow.",
)


if __name__ == "__main__":
    demo.launch()

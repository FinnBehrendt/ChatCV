import threading
from types import SimpleNamespace

from langchain_core.runnables import RunnableLambda

from app.modules import rag_pipeline


def test_qa_initialization_builds_a_valid_retriever_prompt(mocker):
    # Bypass singleton setup so the test cannot create real model clients.
    pipeline = object.__new__(rag_pipeline.ChatRAGPipeline)
    pipeline.qa_chain = None
    pipeline._chain_lock = threading.Lock()
    pipeline.config = SimpleNamespace(
        candidate=SimpleNamespace(name="Example Candidate"),
        chat_system_prompt="Answer questions about {candidate_name}.",
    )
    pipeline.vectorstore_manager = mocker.Mock()
    vectorstore = pipeline.vectorstore_manager.get_vectorstore.return_value
    vectorstore.as_retriever.return_value = mocker.Mock(search_kwargs={})
    pipeline.model_provider = mocker.Mock()
    pipeline.query_validator = mocker.Mock()

    passthrough = RunnableLambda(lambda value: value)
    history_factory = mocker.patch.object(
        rag_pipeline, "create_history_aware_retriever", return_value=passthrough
    )
    mocker.patch.object(
        rag_pipeline, "create_stuff_documents_chain", return_value=passthrough
    )
    mocker.patch.object(
        rag_pipeline, "create_retrieval_chain", return_value=passthrough
    )

    pipeline._initialize_qa_chain()

    history_factory.assert_called_once()
    prompt = history_factory.call_args.kwargs["prompt"]
    messages = prompt.format_messages(chat_history=[], input="Example question")
    assert messages[0].content == "Example question"
    assert messages[-1].content == (
        "given the above conversation, generate a search query "
        "to lookup content important for the conversation."
    )
    assert pipeline.qa_chain is not None

import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed,openai_complete_if_cache,create_openai_async_client
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger,wrap_embedding_func_with_attrs
from lightrag.types import Any
import numpy as np
import json

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage/hotpot-qa"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def openai_embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = "https://api.openai-hub.com/v1",
    api_key: str = "sk-jp2NLH48Z11mQppWVyItB9ZnK8zNW8R0elNhE0naLP7dtjTD",
    client_configs: dict[str, Any] = None,
) -> np.ndarray:
    """Generate embeddings for a list of texts using OpenAI's API.

    Args:
        texts: List of texts to embed.
        model: The OpenAI embedding model to use.
        base_url: Optional base URL for the OpenAI API.
        api_key: Optional OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
        client_configs: Additional configuration options for the AsyncOpenAI client.
            These will override any default configurations but will be overridden by
            explicit parameters (api_key, base_url).

    Returns:
        A numpy array of embeddings, one per input text.

    Raises:
        APIConnectionError: If there is a connection error with the OpenAI API.
        RateLimitError: If the OpenAI API rate limit is exceeded.
        APITimeoutError: If the OpenAI API request times out.
    """
    # Create the OpenAI client
    openai_async_client = create_openai_async_client(
        api_key=api_key, base_url=base_url, client_configs=client_configs
    )

    async with openai_async_client:
        response = await openai_async_client.embeddings.create(
            model=model, input=texts, encoding_format="float"
        )
        return np.array([dp.embedding for dp in response.data])

async def openai_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    result = await openai_complete_if_cache(
        "gpt-4.1-mini",  # context length 128k
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url="https://api.openai-hub.com/v1",
        **kwargs,
    )
    return result

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=openai_complete
    )
    # IMPORTANT: Both initialization calls are required!
    await rag.initialize_storages()  # Initialize storage backends
    await initialize_pipeline_status()  # Initialize processing pipeline
    return rag

async def main():
    rag = None
    try:
        # Initialize RAG instance
        rag = await initialize_rag()
        with open("hotpot-qa/hotpot-document.jsonl") as fp:
            docs = [json.loads(line)["context"] for line in fp]
        docs=docs[:5000]
        # 使用并发异步插入（控制并发数）
        semaphore = asyncio.Semaphore(100)  # 限制并发任务数
        async def insert_doc(doc):
            async with semaphore:
                await rag.ainsert(doc)
                
        # 批量提交任务
        tasks = [insert_doc(doc) for doc in docs]
        await asyncio.gather(*tasks)
        
        print(f"插入完成，共插入 {len(docs)} 个文档")
        # 后续混合搜索逻辑...
        mode = "hybrid"

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
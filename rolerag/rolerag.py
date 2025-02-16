import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast

from .llm import (
    gpt_4o_mini_complete,
    ollama_model_complete,
    ollama_embedding,
    openai_embedding,
)
from .operate import (
    local_query,
    chunking_by_token_size,
    extract_entiti_relation,
    _handle_personality_summarization,
)

from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    DataFrameDBStorage,
    NetworkXStorage,
)

from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    BaseDFrameStorage,
    StorageNameSpace,
    QueryParam,
    SummaryParam,
)

def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_event_loop()

    except RuntimeError:
        logger.info("Creating a new event loop in main thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        return loop

@dataclass
class RoleRAG:
    working_dir: str = field(
        default_factory=lambda: f"./rolerag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    kg: str = field(default="NetworkXStorage")
    source: str = field(default="Beethoven's Biography")

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # text chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 1000

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 3072,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # embedding_func: EmbeddingFunc = field(default_factory=lambda:hf_embedding)
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM
    llm_model_func: callable = gpt_4o_mini_complete  # hf_model_complete#
    llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"  #'meta-llama/Llama-3.2-1B'#'google/gemma-2-2b-it'
    llm_model_max_token_size: int = 10000
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    dataframe_storage_cls: Type[BaseDFrameStorage] = DataFrameDBStorage 
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    enable_llm_cache: bool = True

    # extension
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        log_file = os.path.join(self.working_dir, "rolerag.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"RoleRAG init with param:\n  {_print_config}\n")

        # @TODO: should move all storage setup here to leverage initial start params attached to self.
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class()[
            self.kg
        ]

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)
        
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self),
                embedding_func=None,
            )
            if self.enable_llm_cache
            else None
        )

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self),embedding_func=self.embedding_func
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self), embedding_func=self.embedding_func
        )

        self.alias_mapping_dframe = self.dataframe_storage_cls(
            namespace = "alias_mapping", 
            global_config = asdict(self),
            columns=["alias", "real_name", "description", "chunk_id"],
        )

        self.personality_dframe = self.dataframe_storage_cls(
            namespace = "personality",
            global_config = asdict(self),
            columns = ["name", "personality"]
        )

        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name", "entity_type", "content", "__vector__"},
        )
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id", "description"}, # add more keys.
        )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
                partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache,
                **self.llm_model_kwargs,
            )
        )

    def _get_storage_class(self) -> Type[BaseGraphStorage]:
        return {
            #"Neo4JStorage": Neo4JStorage,
            "NetworkXStorage": NetworkXStorage,
        }

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings):
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]

            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))# keep the keys that have not in the database.
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            inserting_chunks = {}
            for doc_key, doc in new_docs.items():
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in chunking_by_token_size(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                    )
                }
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            logger.info("[Entity/Relationship Extraction]...")
            maybe_new_kg = await extract_entiti_relation(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                alias_mapping=self.alias_mapping_dframe,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities and relationships found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg

            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.entities_vdb,
            self.relationships_vdb,
            self.alias_mapping_dframe,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def summarize_personality(self, characters:list[str], param: SummaryParam = SummaryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.asummary(characters, param))

    async def asummary(self, characters:list[str], param: SummaryParam = SummaryParam()):
        try:
            if isinstance(characters, str):
                characters = [characters]
            
            #Convert to the formal name
            _add_characters = await asyncio.gather(
                    *[self.alias_mapping_dframe.get_column_by_column(name, "alias", "real_name") for name in characters])

            #Check whether the formal name of _add_characters is in the database
            new_characters = await self.personality_dframe.filter_by_column_values(_add_characters, column = "name")

            if not len(new_characters):
                logger.warning("All characters have been summarized in the storage")
                return 
            logger.info(f"Summarizing the personalities of {new_characters} from the knowledge graph")

            await asyncio.gather(
                    *[_handle_personality_summarization(name, 
                        knowledge_graph_inst=self.chunk_entity_relation_graph,
                        relationships_vdb=self.relationships_vdb,
                        personality_db = self.personality_dframe,
                        global_config=asdict(self)) for name in new_characters])

        finally:
            await self._summary_done()

    async def _summary_done(self):
        tasks = []
        for storage_inst in [
            self.personality_dframe,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())

        await asyncio.gather(*tasks)

    def query(self, character:str, source:str, question: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(character, source, question, param))

    async def aquery(self, character:str, source:str, question: str, param: QueryParam = QueryParam()):
        try:
            response = await local_query(
                character,
                source,
                question,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                self.alias_mapping_dframe,
                param,
                asdict(self),
            )
        finally: 
            await self._query_done()
        return response

    async def _query_done(self):
        logger.info("Query Finished!")

    def delete_by_entity(self, entity_name: str):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        entity_name = f'"{entity_name.upper()}"'

        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.relationships_vdb.delete_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self):
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

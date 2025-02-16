import asyncio
import json
import re
import random
import ast
import networkx as nx
import numpy as np
from copy import deepcopy
from scipy.spatial.distance import cdist
from typing import Union
from itertools import combinations
from collections import Counter, defaultdict
import warnings
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    BaseDFrameStorage,
    TextChunkSchema,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS

def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results

async def _handle_relation_summary(
    relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    #if len(tokens) < summary_max_tokens:  # No need for summary
    #    return description
    prompt_template = PROMPTS["summarize_relation_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        relation_name=relation_name,
        relationship_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary

async def _handle_entity_summary(
    entity_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    #if len(tokens) < summary_max_tokens:  # No need for summary
    #    return description

    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary

async def _handle_identity_verification(
    src_name:str,
    src_content:str,
    tgt_name:str,
    tgt_content:str,
    global_config:dict,
) -> str:

    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]

    prompt_template = PROMPTS['identity_verification']
    use_prompt = prompt_template.format(src_name = src_name, 
            src_content = src_content,
            tgt_name = tgt_name,
            tgt_content = tgt_content,
            source = global_config['source'])

    response = await use_llm_func(use_prompt, max_tokens = llm_max_tokens)
    #parse results with regex expression.
    match = re.search(r"answer:\s*(.*)", response, re.IGNORECASE)

    count = 0
    #We try 5 times, if still failing to giving answers, we return "no"
    while count < 5:
        if match:
            answer = match.group(1)
            if answer.lower() in ["yes", "no"]:
                break

        response = await use_llm_func(use_prompt, max_tokens = llm_max_tokens)
        match = re.search(r"answer:\s*(.*)", response, re.IGNORECASE)
        count += 1

    if count < 5:
        return answer
    else:
        return "no"

async def _extract_formal_entity(
    name_list:list[str],
    global_config:dict,
) -> str:

    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]

    prompt_template = PROMPTS['formal_name']
    use_prompt = prompt_template.format(name_list=name_list, source = global_config['source'])
    response = await use_llm_func(use_prompt, max_tokens = llm_max_tokens)
    
    #Parse results with regex expression.
    match = re.search(r"Most Popular Name:\s*(.*)", response, re.IGNORECASE)
    count = 0
    #We try 5 times, if still failing to giving answers, we return "no"

    while count < 5:
        if match:
            answer = match.group(1).strip()
            if answer in name_list:
                break

        response = await use_llm_func(use_prompt, max_tokens = llm_max_tokens)
        match = re.search(r"Most Popular Name:\s*(.*)", response, re.IGNORECASE)
        count += 1
    
    print("The most popular name is {} of {}".format(answer, name_list))
    if count < 5:
        return answer
    else:
        return name_list[0]

async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != 'entity':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )

async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != 'relationship':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_weight = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(edge_weight) if is_float_regex(edge_weight) else 1.0
    )

    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )

async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = Counter([dp["entity_type"] for dp in nodes_data] + already_entitiy_types).most_common()[0][0]

    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    if len(nodes_data) > 1 or already_node is not None:
        description = await _handle_entity_summary(
            entity_name, description, global_config
        )

    node_data = dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])

    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": 'UNKNOWN',
                },
            )
    description = await _handle_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
    )

    return edge_data

async def _handle_identity_merging(
        maybe_nodes,
        entity_vdb:BaseVectorStorage,
        alias_mapping:BaseDFrameStorage,
        global_config:dict,
    ) ->defaultdict(list):

    embedding_func = global_config['embedding_func']
    for extracted_name, entity_list in maybe_nodes.items():
        current_node = dict()
        current_node["entity_name"] = extracted_name
        entity_description = list()
        chunk_id_list = list()
        entity_type = list()
        for index, entity in enumerate(entity_list):
           entity_type.append(entity['entity_type'])
           entity_description.append(entity['description'])
           chunk_id_list.append(entity['source_id'])
        
        # sumarize entity description
        if len(entity_description) > 1:
            description = GRAPH_FIELD_SEP.join(entity_description)
            description_summary = await _handle_entity_summary(extracted_name, description, global_config)
            current_node["content"] = description_summary
        else:
            description_summary = entity_description[0]
            current_node["content"] = description_summary

        embedding = await embedding_func(extracted_name + description_summary)
        source_id_str = GRAPH_FIELD_SEP.join(chunk_id_list)

        if isinstance(embedding, list):
            embedding = np.array(embedding)
            current_node['__vector__'] = embedding.mean(0)
        else:
            current_node['__vector__'] = embedding.mean(0)

        current_node["entity_type"] = Counter(entity_type).most_common()[0][0]
        flag = await entity_vdb.isin(extracted_name)

        #update maybe_nodes 
        copied_current_node = deepcopy(current_node)
        copied_current_node.pop("__vector__")
        copied_current_node["description"] = copied_current_node.pop("content")
        copied_current_node["source_id"] = source_id_str
        maybe_nodes[extracted_name] = [copied_current_node]

        if flag: # current entity has already been in the entity database.
            continue
        else:
            top_k_nodes = await entity_vdb.query_embed(embedding[0], top_k = 3)
            tmp_nodes = dict()
            tmp_nodes[extracted_name] = current_node
            # if not in, then add the node into the vdb
            await entity_vdb.upsert_embed(tmp_nodes)

            if len(top_k_nodes) > 0:
                """
                results = await asyncio.gather(
                        *[_handle_identity_verification(extracted_name, description_summary, entity['entity_name'], entity['content'], global_config) 
                          if entity['entity_type'].lower() != "time" and current_node["entity_type"].lower() != "time" else 
                    _handle_identity_verification(extracted_name, "Unit of Time", entity['entity_name'], "Unit of Time", global_config)
                          for entity in top_k_nodes]
                    )
                """

                results = [] 
                for entity in top_k_nodes:
                    if entity['entity_type'] != current_node['entity_type']:
                        results.append("no")
                    elif entity['entity_type'].lower() == "time" and current_node['entity_type'].lower() == "time":
                        result = await _handle_identity_verification(extracted_name, "Time", entity['entity_name'], "Time", global_config)
                        results.append(result)
                    else:
                        result = await _handle_identity_verification(extracted_name, description_summary, entity['entity_name'], entity['content'], global_config)
                        results.append(result)

                for i in range(len(top_k_nodes)):
                    if top_k_nodes[i]['entity_type'] == current_node['entity_type']:
                        if results[i].lower() == "yes":
                            alias = dict(
                                alias=extracted_name,
                                real_name=top_k_nodes[i]['entity_name'],
                                description=description_summary,
                                chunk_id=chunk_id_list)

                            await alias_mapping.upsert([alias])

    return

async def _handle_entity_normalization(
    alias_mapping: BaseDFrameStorage,
    global_config:dict
):

    def find_name_groups(df):
        # Create a graph
        graph = nx.Graph()
        # Add edges from the alias-same_name pairs
        for _, row in df.iterrows():
            graph.add_edge(row['alias'], row['real_name'])

        # Find connected components
        groups = [list(component) for component in nx.connected_components(graph)]
        return groups
    
    groups = find_name_groups(alias_mapping._data)
    results = await asyncio.gather(
            *[_extract_formal_entity(group, global_config)
                for group in groups]
            )

    #rename the dataframe
    for i in range(len(groups)):
        group = groups[i]
        formal_name = results[i]
        # If formal_name -> alias, then we switch them.
        mask = (alias_mapping._data['alias'] == formal_name)
        alias_mapping._data.loc[mask, ['alias', 'real_name']] = alias_mapping._data.loc[mask, ['real_name', 'alias']].values
        
        #replace xx in the alias -> xx with the formal name
        group.remove(formal_name)
        #alias_mapping.replace('alias', group, 'real_name', formal_name)
        alias_mapping._data.loc[alias_mapping._data['alias'].isin(group), 'real_name'] = formal_name

async def rename_entity_with_alias(
    maybe_nodes: defaultdict(list),
    maybe_edges: defaultdict(list),
    alias_mapping: BaseDFrameStorage,
):

    def name_mapping(src:str) -> str:
        try:
            real_name = alias_mapping._data.loc[alias_mapping._data['alias'] == src, 'real_name'].iat[0]
            if real_name is not None:
                return real_name
            else:
                return src
        except:
            return src
    
    nodes_to_remove = []
    nodes_to_add = []

    for name, entity in maybe_nodes.items():
        real_name = name_mapping(name)
        if real_name != name:
            nodes_to_remove.append(name)
            nodes_to_add.append(real_name)

    for i, name in enumerate(nodes_to_remove):
        src_ent = maybe_nodes[name]
        # update the name of entity to the formal name.
        for j in range(len(src_ent)):
            src_ent[j]['entity_name'] = nodes_to_add[i]

        maybe_nodes[nodes_to_add[i]].extend(src_ent)
        del maybe_nodes[name]
    
    edges_to_remove = []
    edges_to_add = []

    for edge_name, edges in maybe_edges.items():
        src_id = edge_name[0]
        tgt_id = edge_name[1]
        real_src_id = name_mapping(src_id)
        real_tgt_id = name_mapping(tgt_id)
        if edge_name != (real_src_id, real_tgt_id):
            edges_to_add.append((real_src_id, real_tgt_id))
            edges_to_remove.append(edge_name)

    for i, edge_name in enumerate(edges_to_remove):
        src_edge = maybe_edges[edge_name]
        tgt_edge_name = edges_to_add[i]
        for j in range(len(src_edge)):
            src_edge[j]['src_id'] = tgt_edge_name[0]
            src_edge[j]['tgt_id'] = tgt_edge_name[1]
        
        maybe_edges[tgt_edge_name].extend(src_edge)
        del maybe_edges[edge_name]
    
async def _handle_personality_summarization(
        name: str,
        knowledge_graph_inst: BaseGraphStorage,
        relationships_vdb: BaseVectorStorage,
        personality_db: BaseGraphStorage,
        global_config: dict,
    ) -> None:
    
    #Extract description
    already_node = await knowledge_graph_inst.get_node(name)
    descriptions = list()
    if already_node is not None:
        already_description = already_node["description"]
        descriptions.extend(already_description.split(GRAPH_FIELD_SEP))

    use_llm_func: callable = global_config["llm_model_func"]
    personality_summary_prompt = PROMPTS["personality_summary"]
    hint_prompt = personality_summary_prompt.format(descriptions = descriptions)
    final_results = await use_llm_func(hint_prompt)

    name_personality = dict(
        name = name,
        personality=final_results
        )
    await personality_db.upsert([name_personality])

async def extract_entiti_relation(
        chunks: dict[str, TextChunkSchema],
        knowledge_graph_inst: BaseGraphStorage,
        entity_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        alias_mapping: BaseDFrameStorage,
        global_config: dict,
    ) -> Union[BaseGraphStorage, None]:

    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())

    entity_extract_prompt = PROMPTS["entiti_relation_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )

    entity_types = PROMPTS["DEFAULT_ENTITY_TYPES"]
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        entiti_type_wrong = []

        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                if if_entities['entity_type'].lower() in entity_types:
                    maybe_nodes[if_entities["entity_name"]].append(if_entities)
                else:
                    entiti_type_wrong.append(if_entities["entity_name"]) # remove entities out of scope
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                if if_relation["src_id"] not in entiti_type_wrong and if_relation["tgt_id"] not in entiti_type_wrong:
                    maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                        if_relation
                    )
                continue
            
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)

        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    
    print() # clear the progress bar
    maybe_nodes = defaultdict(list) # clear the progress bar
    
    print()  # clear the progress bar
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)

    # merge identities with the same name or alias.
    await _handle_identity_merging(maybe_nodes, entity_vdb, alias_mapping, global_config)
    # Map the alias with the formal name
    await _handle_entity_normalization(alias_mapping, global_config)
    # Rename
    await rename_entity_with_alias(maybe_nodes, maybe_edges, alias_mapping)
    
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    all_relationships_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )

    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if not len(all_relationships_data):
        logger.warning(
            "Didn't extract any relationships, maybe your LLM is not working"
        )
        return None

    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content":dp["src_id"]
                + dp["tgt_id"]
                + dp["description"],
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst

async def _build_local_summarization(
    entity_list:list[str],
    knowledge_graph_inst:BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    alias_mapping: BaseDFrameStorage,
    global_config: dict
) -> str:

    entity_description = {}
    
    #extract entity description within the database.
    for entity in entity_list:
        already_node = await knowledge_graph_inst.get_node(entity)
        if already_node is not None:
            entity_description[entity] = already_node["description"]
    
    #extract the relations.
    elements = list(entity_description.keys())
    combines_of_entiti = list(combinations(elements, 2))
    entiti_relations = []
    for entiti_pair in combines_of_entiti:
        already_edge = await knowledge_graph_inst.get_edge(entiti_pair[0], entiti_pair[1])
        if already_edge is not None:
            entiti_relations.append(already_edge["description"])

    context_prompt_temp = PROMPTS["context_summarization"]
    context_prompt = context_prompt_temp.format(entity_description = entity_description, entiti_relations = entiti_relations)
    use_llm_func: callable = global_config["llm_model_func"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    summary = await use_llm_func(context_prompt, max_tokens=summary_max_tokens)

    return summary

async def parse_keywords(character, final_result:str, context_base):

    records = split_string_by_multi_markers(final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]])

    keywords = list()
    specific_keywords = list()
    general_keywords = list()
    unrelated_keywords = list()

    for record in records:
        record = re.search(r"\((.*)\)", record)
        if record is None:
            continue
        record = record.group(1)
        record_attributes = split_string_by_multi_markers(
            record, [context_base["tuple_delimiter"]]
        )
        if len(record_attributes) < 5:
            continue
        else:
            name = record_attributes[1]
            type = record_attributes[0]
            analysis = record_attributes[2]
            keywords.append(dict(entity_name=name, entity_type=type))
            familarity = record_attributes[3]
            level = record_attributes[4]
            if familarity.lower() == "no":
                unrelated_keywords.append(dict(entity_name=name, entity_type=type, analysis=analysis))

            if familarity.lower() == "yes":
                if level.lower() == "general":
                    general_keywords.append(dict(entity_name=name, entity_type=type, analysis=analysis))
                if level.lower() == "specific":
                    specific_keywords.append(dict(entity_name=name, entity_type=type, analysis=analysis))
    
    return general_keywords, specific_keywords, unrelated_keywords

async def local_query(
    character: str,
    source: str,
    question: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    alias_mapping_dframe: BaseDFrameStorage,
    query_param,
    global_config: dict,
) -> str:

    context = None
    use_model_func = global_config["llm_model_func"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )

    entity_types = PROMPTS["DEFAULT_ENTITY_TYPES"]
    question = "{}: \t {}".format(character, question)
    kw_prompt_temp = PROMPTS["keyword_extractionv2"]
    #continue_prompt = PROMPTS["keyword_continue_extraction"]
    kw_prompt = kw_prompt_temp.format(**context_base, question = question, character = character, source = source)
    final_result = await use_model_func(kw_prompt)
    
    """
    history = pack_user_ass_to_openai_messages(kw_prompt, final_result)
    for i in range(1):
        glean_result = await use_model_func(continue_prompt, history_messages = history)
        history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
        final_result = glean_result
    """
    
    print(final_result)
    general_keywords, specific_keywords, unrelated_keywords = await parse_keywords(character, final_result, context_base)
    
    count = 0
    while len(general_keywords) == 0 and len(unrelated_keywords) == 0 and len(specific_keywords) == 0:
        final_result = await use_model_func(kw_prompt)
        general_keywords, specific_keywords,  unrelated_keywords = await parse_keywords(character, final_result, context_base)
        count += 1
        if count > 5:
            return ""

    retrieve_nodes = list()
    retrieve_relations = list()
    retrieve_chunks = list()
    irrelative_info = list()

    specific_keywords = specific_keywords[1:] #Remove the speaker itself.
    #print(specific_keywords)
    
    #Retrieve information based on specific entities. The speaker itself is a specifc keyword.
    if len(specific_keywords) > 0:
        use_nodes_s, use_relations_s, use_text_units_s = await _build_local_query_context(
            character,
            specific_keywords,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            alias_mapping_dframe,
            query_param,
        )

        retrieve_nodes.extend(use_nodes_s)
        retrieve_relations.extend(use_relations_s)
        retrieve_chunks.extend(use_text_units_s)

    #Retrive entities having the same type as the general entities.
    if len(general_keywords) > 0:
        use_nodes_g, use_relations_g, use_text_units_g = await _build_general_query_context(
            character,
            specific_keywords, # Specific keywords here are just used to check whether the target character information has been retrieved.
            general_keywords,
            knowledge_graph_inst,
            text_chunks_db,
            alias_mapping_dframe,
            query_param,
        )
        if len(use_nodes_g) > 0:
            retrieve_nodes.extend(node for node in use_nodes_g if node not in retrieve_nodes)
        
        if len(use_relations_g) > 0:
            retrieve_relations.extend(edge for edge in use_relations_g if edge not in retrieve_relations)

        if use_text_units_g is not None:
            retrieve_chunks.extend(chunk for chunk in use_text_units_g if chunk not in retrieve_chunks)
    
    if len(specific_keywords) == 0 and len(general_keywords) == 0:

        alias_table = alias_mapping_dframe._data
        try:
            character = alias_table[alias_table['alias'] == character.upper()]['real_name'].iloc[0]
        except:
            character = character.upper()

        character_node = await knowledge_graph_inst.get_node(character)
        use_text_units = await _find_most_related_text_unit_from_entities(
             [character_node], query_param, text_chunks_db
          )
        retrieve_nodes.append(character_node)
        retrieve_chunks.extend(use_text_units)

    #Reject irrelative information
    if len(unrelated_keywords) > 0:
        irrelative_info = await _build_unrelated_context(character, unrelated_keywords)

    context = await format_content(retrieve_nodes, retrieve_relations, retrieve_chunks, irrelative_info)

    if context is None:
        context = ""

    if query_param.only_need_context:
        return context
    else:
        prompt_temp = PROMPTS["single_response_en"]
        use_prompt = prompt_temp.format(
            **context_base,
            context_data=context,
            character = character,
            question = question,
        )
        response = await use_model_func(
            use_prompt,
        )
        return response


async def _build_unrelated_context(
    character: str,
    unrelated_keywords: defaultdict(list)
):

    irrelative_info = list()
    for keyword in unrelated_keywords:
        entity_name = keyword['entity_name']
        irrelative_info.append([character, entity_name, keyword['analysis']])

    return irrelative_info

async def _build_general_query_context(
    character: str,
    specific_keywords: defaultdict(list),
    general_keywords: defaultdict(list),
    knowledge_graph_inst: BaseGraphStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    alias_mapping_dframe: BaseDFrameStorage,
    query_param,
):

    #retrieve character's profile
    alias_table = alias_mapping_dframe._data
    try:
        character = alias_table[alias_table['alias'] == character.upper()]['real_name'].iloc[0]
    except:
        character = character.upper()

    character_node = await knowledge_graph_inst.get_node(character)

    #type of general entities. The entity types are used to retrieve relative nodes.
    general_types = list()
    for keyword in general_keywords:
        general_types.append(keyword['entity_type'])

    #Get edges of character node
    edges = await knowledge_graph_inst.get_node_edges(character)

    all_one_hop_nodes = list()
    for e in edges:
        all_one_hop_nodes.append(e[1])

    all_one_hop_nodes_data = await asyncio.gather(
            *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes])

    _1hop_nodes_data = []
    for node in all_one_hop_nodes_data:
        if node['entity_type'].lower() in general_types:
            _1hop_nodes_data.append(node)

    #Get edges
    all_related_edges = list()
    for node in _1hop_nodes_data:
        edge_data = await knowledge_graph_inst.get_edge(character, node["entity_name"])
        edge_data['src_id'] = character
        edge_data['tgt_id'] = node["entity_name"]
        all_related_edges.append(edge_data)

    #Sorted by edge weight and fetch top-K relevant nodes and edges.
    edge_weight = list()
    for edge in all_related_edges:
        edge_weight.append(edge['weight'])

    idx = np.argsort(edge_weight)[::-1]
    idx = idx[0:query_param.top_k]
    #retrieve node information.
    if len(specific_keywords) == 0:
        use_nodes = [character_node]
        use_text_units = await _find_most_related_text_unit_from_entities(
            use_nodes, query_param, text_chunks_db
        )
    else:
        use_nodes = []

    use_relations = []
    _min = min(query_param.top_k, len(all_related_edges))
    for i in range(_min):
        use_nodes.append(_1hop_nodes_data[idx[i]])
        use_relations.append(all_related_edges[idx[i]])

    if len(specific_keywords) == 0:
        return use_nodes, use_relations, use_text_units
    else:
        return use_nodes, use_relations, None

async def _build_local_query_context(
    character: str,
    keywords: defaultdict(list),
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    alias_mapping_dframe: BaseDFrameStorage,
    query_param,
):

    #retrieve character's profile
    alias_table = alias_mapping_dframe._data
    try:
        character = alias_table[alias_table['alias'] == character.upper()]['real_name'].iloc[0]
    except:
        character = character.upper()

    character_node = await knowledge_graph_inst.get_node(character)
    relative_nodes = [character_node]

    # First, we try to search the node by names in graph.
    for entity in keywords:
        try:
            real_name = alias_table[alias_table['alias'] == entity['entity_name']]['real_name'].iloc[0]
        except:
            real_name = entity['entity_name']

        if await knowledge_graph_inst.has_node(real_name):
            node = await knowledge_graph_inst.get_node(real_name)
            relative_nodes.append(node)
            keywords.remove(entity)

    #retrieve similar entities of specific entities
    entities_in_vdb = await asyncio.gather(
            *[entities_vdb.query(keyword["entity_name"], 0.55, 3)
              for keyword in keywords])

    entities_in_graph = defaultdict(list)
    #map the entiti to the formal name via alias_mapping form
    for i in range(len(entities_in_vdb)):
        entities_in_graph[keywords[i]['entity_name']] = []
        for j in range(len(entities_in_vdb[i])):
            entity = entities_in_vdb[i][j]
            if entity["entity_type"].lower() == keywords[i]['entity_type'].lower():
                try:
                    #Obtain the real name in graph, the entity vdb contains multiple names of a specific role.
                    real_name = alias_table[alias_table['alias'] == entity['entity_name']]['real_name'].iloc[0]
                except:
                    real_name = entity['entity_name']
                # some entities in vdb may map to the same entity in the graph.
                if real_name not in entities_in_graph[keywords[i]['entity_name']]:
                    entities_in_graph[keywords[i]['entity_name']].append(real_name)

    #irrelative_info = [["src_id", "tgt_id", "description"]]
    if len(entities_in_graph) > 0:
        for name, entities in entities_in_graph.items():
            if len(entities) > 0:
                node_datas = await asyncio.gather(
                    *[knowledge_graph_inst.get_node(en) for en in entities]
                )
                relative_nodes.extend(node_datas)

    if not all([n is not None for n in relative_nodes]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    #Retrieve the most relative chunk by entity IDs.
    use_text_units = await _find_most_related_text_unit_from_entities(
        relative_nodes, query_param, text_chunks_db
        )

    #Retrieve relations from graph
    use_relations = await _find_most_related_edges_from_entities(
            relative_nodes, query_param, knowledge_graph_inst

        )

    return relative_nodes, use_relations, use_text_units

async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    flat_list = [item for sublist in text_units for item in sublist]
    counter = Counter(flat_list)
    max_count = counter.most_common(1)[0][1]
    most_units = []
    for unit, count in counter.items():
        if count == max_count:
            most_units.append(unit)
    
    chunk_data = await asyncio.gather(
            *[text_chunks_db.get_by_id(c_id) for c_id in most_units])
    
    all_text_units = truncate_list_by_token_size(
        chunk_data,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    #all_text_units = GRAPH_FIELD_SEP.join([t["content"] for t in all_text_units])
    return all_text_units

async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param,
    knowledge_graph_inst: BaseGraphStorage,
):
    source_node = node_datas[0]
    target_nodes = node_datas[1:]
    
    src_id = source_node['entity_name']
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(src_id, dp['entity_name']) for dp in target_nodes])
    
    for i in range(len(all_related_edges)):
        if all_related_edges[i] is not None:
            all_related_edges[i]["src_id"] = src_id
            all_related_edges[i]["tgt_id"] = target_nodes[i]["entity_name"]

    all_related_edges = [x for x in all_related_edges if x is not None]

    all_edges_data = sorted(
        all_related_edges, key=lambda x: x["weight"], reverse=True
    )
    
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    return all_edges_data

async def format_content(
    retrieve_nodes,
    retrieve_relations,
    retrieve_text_units,
    irrelative_info,
):
    # Text Chunk
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(retrieve_text_units):
        text_units_section_list.append([i, t["content"]])

    text_units_context = list_of_list_to_csv(text_units_section_list)

    #Retrive Entity Information.
    entites_section_list = [["entity", "type", "description"]]
    for i, n in enumerate(retrieve_nodes):
        entites_section_list.append(
            [
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),

            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    # Retrieve Relationships
    relations_section_list = [["source", "target", "description"]]
    for i, e in enumerate(retrieve_relations):
        relations_section_list.append(
            [
                e["src_id"],
                e["tgt_id"],
                e["description"],
            ]
        )
    for i, e, in enumerate(irrelative_info):
        relations_section_list.append(e)

    relations_context = list_of_list_to_csv(relations_section_list)

    ## End Retrieval 
    #```csv{text_units_context}```
    logger.info(
        f"Local query uses {len(retrieve_nodes)} entites, {len(retrieve_relations)} relations, {len(retrieve_text_units)} chunks"
    )
    return f"""
        -----Chunks-----
        ```csv
        {text_units_context}
        ```
        -----Entities-----
        ```csv
        {entities_context}
        ```
        -----Relationships-----
        ```csv
        {relations_context}
        ```
    """

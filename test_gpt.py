import os
from typing import Optional
from rolerag import RoleRAG
from rolerag.llm import gpt_4o_mini_complete, ollama_model_complete, ollama_embedding, gpt_4o_complete
from rolerag.utils import EmbeddingFunc

WORKING_DIR = "workspace"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

def remove_all_files_in_path(path):
    """
    Remove all files under the given path if the path is not empty.
    :param path: Path to the directory
    """
    if os.path.exists(path) and os.path.isdir(path):  # Ensure the path exists and is a directory
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            try:
                os.remove(file_path)  # Remove the file
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")
        print(f"All files under {path} have been removed.")
    else:
        print(f"The path {path} does not exist or is not a directory.")

remove_all_files_in_path(WORKING_DIR) # this function is just used under the debug stage.

rag = RoleRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
        source = "Beethoven's Biography"
)

with open("./corpus/Beethoven.txt") as f:
    rag.insert(f.read())


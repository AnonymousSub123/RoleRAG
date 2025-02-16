# RoleRAG
Source codes for Rolerag

## Installation

```
python setup.py install
```

## Create Graph
```
import os
from rolerag import RoleRAG
from rolerag.llm import gpt_4o_mini_complete
WORKING_DIR = "workspace"

rag = RoleRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
        source = "Beethoven's Biography"
)

with open("./corpus/Beethoven.txt") as f:
    rag.insert(f.read())

```

import os
from rolerag import RoleRAG
from rolerag.llm import gpt_4o_mini_complete
#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

WORKING_DIR = "workspace"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = RoleRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)
print(rag.query("BEETHOVEN", "What is your interest?"))
#print(rag.query("BEETHOVEN", "what do you think of your father?"))
#print(rag.query("BEETHOVEN", "How has your experience with deafness affected your approach to music composition?"))
#print(rag.query("BEETHOVEN", "I want to learn C++, could you please recommand some books for me?"))
print(rag.query("BEETHOVEN", "How do you write a quick sort in Python?"))
print("\n" * 3)
"""
print("\n" * 3)
print(rag.query("LUDWIG VAN BEETHOVEN", "Talk about the car you drove yesterday?"))
print("\n" * 3)
print(rag.query("LUDWIG VAN BEETHOVEN", "How do you write a quick sort in Python?"))
print("\n" * 3)
print(rag.query("LUDWIG VAN BEETHOVEN", "I want to learn C++, could you please recommand some books for me?"))
"""

from fastapi import FastAPI, Query
from typing import List, Optional
from annotator import propose
from program import DafnyProgram
from search import SearchNode, VLLMProposer

proposer = VLLMProposer("gpoesia/dafny-annotator-8B", "meta-llama/Llama-3.1-8B")

app = FastAPI()

@app.post("/annotate")
async def annotate(program: str) -> List[str]:
    node = SearchNode(DafnyProgram(program))
    result = proposer.propose([node])
    return result[0]

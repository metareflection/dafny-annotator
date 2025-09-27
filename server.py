from fastapi import FastAPI
from typing import List, Optional
from annotator import propose
from program import DafnyProgram
from search import SearchNode, VLLMProposer, batch_greedy_search

proposer = VLLMProposer("gpoesia/dafny-annotator-8B", "meta-llama/Llama-3.1-8B")

app = FastAPI()

@app.post("/annotate")
async def annotate(program: str) -> List[str]:
    node = SearchNode(DafnyProgram(program))
    results = proposer.propose([node])
    return results[0]

@app.post("/greedy_search")
async def search(program: str) -> Optional[str]:
    programs = [DafnyProgram(program)]
    results = batch_greedy_search(programs, proposer, 5, None)
    return results[0]

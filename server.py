from fastapi import FastAPI
from typing import List, Optional
from annotator import propose
from program import DafnyProgram
from search import SearchNode, VLLMProposer, batch_greedy_search
import os

model_path = os.environ.get("MODEL_PATH", "gpoesia/dafny-annotator-8B")
proposer = VLLMProposer(model_path, "meta-llama/Llama-3.1-8B")

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
    result = results[0]
    return str(result) if result else None

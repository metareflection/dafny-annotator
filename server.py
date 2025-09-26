from transformers import AutoTokenizer, AutoModelForCausalLM
#from peft import AutoPeftModelForCausalLM
from fastapi import FastAPI, Query
from typing import List, Optional
from annotator import propose
from program import DafnyProgram
from test_example import verification_prompt
from synchromesh import HuggingFaceModel

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("gpoesia/dafny-annotator-8B", device_map="auto")
#model = AutoPeftModelForCausalLM.from_pretrained("gpoesia/finetuned_Meta-Llama-3.1-8B_dafnybench-100-peft", device_map="auto")
lm = HuggingFaceModel(model, tokenizer=tokenizer, prompt_template=verification_prompt, temperature=1)

app = FastAPI()

@app.post("/annotate")
async def annotate(program: str, num_samples: Optional[int] = Query(None, description="Number of samples (default to 1)")) -> List[str]:
    r = propose(lm, DafnyProgram(program), num_samples or 1)
    return [str(p) for p in r]

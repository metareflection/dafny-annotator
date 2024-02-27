from dataclasses import dataclass, field
from transformers import HfArgumentParser, set_seed
@dataclass
class Arguments:
    model: str = field(default="Phind/Phind-CodeLlama-34B-v2", metadata={"help": "HuggingFace Model"})
def get_args():
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]
    return args
args = get_args()

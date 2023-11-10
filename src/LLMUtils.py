import llama_cpp
import multiprocessing as mp
from dataclasses import field
from typing import List

#  Explanation for these parameters is found at https://abetlen.github.io/llama-cpp-python/
class LLMConfig:
    def __init__(self):
        self.ctxParams = llama_cpp.llama_context_default_params()

        self.modelParams = llama_cpp.llama_model_default_params()
        self.modelPath = None
        self.modelName = ""
        self.modelType = "" # remove, or not?

        self.nOffloadLayer = 0
        self.mainGPU = 0
        self.threads = mp.cpu_count()

        self.newlineToken = '\n'
        self.EOT = '\n'
        self.prompt: str = ""
        self.antiPrompt: List[str] = [] # field(default_factory=list)

        self.nCtx: int = 512
        self.n_keep: int = 256
        self.n_predict: int = 128
        self.tfs_z: float = 1.00
        self.typical_p: int = 1
        self.top_k: int = 40
        self.top_p: float = 0.90
        self.temperature: float = 0.6
        self.mirostat: int = 2
        self.mirostat_tau: float = 5.0
        self.mirostat_eta: float = 0.1
        self.repeat_last_n: int = 64
        self.repeat_penalty: int = 1
        self.frequency_penalty: int = 0
        self.presence_penalty: int = 0
        self.penalize_nl: bool = False
        self.n_batch: int = 24

        self.logitsAll = False
        self.logit_bias: dict[int, float] = field(default_factory=dict)

    def getCtxParms(self):
        return self.ctxParams

    def getModelParams(self):
        return self.modelParams

def defaultLlamactxParams():
    return LLMConfig()
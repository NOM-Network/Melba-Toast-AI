import llama_cpp
from dataclasses import dataclass, field
from typing import List

class LLMConfig:
    def __init__(self):
        self.ctxParams = llama_cpp.llama_context_default_params()
        self.modelParams = llama_cpp.llama_model_default_params()
        self.modelPath = None
        self.modelName = ""
        self.modelType = "" # remove
        self.nOffloadLayer = 10
        self.mainGPU = 0

        self.newlineToken = '\n'
        self.EOT = '\n'

        self.prompt: str = ""
        self.antiPrompt: List[str] = [] # field(default_factory=list)

        self.n_keep: int = -1
        self.n_predict: int = 256
        self.tfs_z: float = 1.00
        self.typical_p: int = 1
        self.top_k: int = 40
        self.top_p: float = 0.90
        self.temperature: float = 0.6
        self.mirostat: int = 0
        self.mirostat_tau: float = 5.0
        self.mirostat_eta: float = 0.1
        self.repeat_last_n: int = 64
        self.repeat_penalty: int = 1
        self.frequency_penalty: int = 0
        self.presence_penalty: int = 0
        self.penalize_nl: bool = False
        self.n_batch: int = 24

        self.logit_bias: dict[int, float] = field(default_factory=dict)

    def getCtxParms(self):
        return self.ctxParams

    def getModelParams(self):
        return self.modelParams

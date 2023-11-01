import LLMUtils
import llama_cpp
from llama_cpp import Llama
import numpy as np
import numpy.typing as npt
from typing import List
from time import time
from random import randint

def defaultLlamactxParams():
    return LLMUtils.LLMConfig()

class LlamaModel:
    def __init__(self, parameters: LLMUtils.LLMConfig):
        self.parameters = parameters
        self.threadCount = self.parameters.threads
        if self.threadCount < 6:
            print("Low thread count. Inference might be slow.")

        self.ctxParams = self.parameters.getCtxParms()
        self.ctxParams.n_ctx = self.parameters.nCtx  # default
        self.ctxParams.n_threads = self.threadCount
        if self.ctxParams.seed <= 0:
            self.ctxParams.seed = int(randint(0, int(time())))

        self.modelParams = self.parameters.getModelParams()
        self.modelParams.n_gpu_layers = self.parameters.nOffloadLayer
        self.modelParams.main_gpu = self.parameters.mainGPU
        self.modelPath = self.parameters.modelPath
        print(str(self.modelParams.n_gpu_layers) + " " + str(self.modelParams.main_gpu))

        if self.modelPath is not None:
            try:
                self.model = llama_cpp.llama_load_model_from_file(self.modelPath.encode('utf-8'), self.modelParams)
            except FileExistsError:
                print("Invalid filepath for model")
        else:
            self.warnAndExit(function="__init__", errorMessage="No model path found")

        if self.model:
            self.context = llama_cpp.llama_new_context_with_model(self.model, self.ctxParams)

        self.n_ctx = llama_cpp.llama_n_ctx(self.context)
        self.nVocab = llama_cpp.llama_n_vocab(self.model) if self.model else self.warnAndExit("__init__",
                                                                                              "No model was found")
        self.nCtx = llama_cpp.llama_n_ctx(self.context) if self.context else self.warnAndExit("__init__",
                                                                                              "No context was found")
        self._init()

    def _init(self):
        self.pCandidatesData = np.array(
            [],
            dtype=np.dtype(
                [("id", np.intc), ("logit", np.single), ("p", np.single)],
                align=True
            ),
        )

        self.pCandidatesData.resize(3, self.nVocab, refcheck=False)
        candidates = llama_cpp.llama_token_data_array(
            data=self.pCandidatesData.ctypes.data_as(llama_cpp.llama_token_data_p),
            size=self.nVocab,
            sorted=False
        )

        self.pCandidates = candidates
        self.EOSToken = llama_cpp.llama_token_eos(self.context)
        self.pCandidatesDataId = np.arange(self.nVocab, dtype=np.intc)
        self.pCandidatesDataP = np.zeros(self.nVocab, dtype=np.single)
        self.pastTokens = 0

        self.inputIds: npt.NDArray[np.intc] = np.ndarray((self.nCtx,), dtype=np.intc)
        self.scores: npt.NDArray[np.single] = np.ndarray((self.nCtx, self.nVocab), dtype=np.single)

    def reset(self):
        self._init()

    def update(self, newParameters: LLMUtils.LLMConfig):
        self.parameters = newParameters
        self.nCtx = self.parameters.nCtx

    def warnAndExit(self, function, errorMessage):
        raise RuntimeError(f"LLMCore: Error in function: '{function}'. Following error message was provided: '{errorMessage}'\n")

    def tokenizeFull(self, input: str, bos: bool = False) -> List[int]:   # Possibly further abstract by adding single
        tokens = (llama_cpp.llama_token * self.nCtx)()              # token, tokenization
        newTokens = llama_cpp.llama_tokenize(model=self.model,
                                             text=input.encode("utf8"),
                                             text_len=len(input.encode("utf8")),
                                             tokens=tokens,
                                             n_max_tokens=self.nCtx,
                                             add_bos=bos)
        #print(len(input.encode("utf8")))
        print(f"LLMCore: {newTokens} token(s) were tokenized.")
        return list(tokens[:newTokens])

    def evaluate(self, tokens: List[int], batch: int):
        if batch > 0:
            for i in range(0, len(tokens), batch):
                nBatch = tokens[i : min(len(tokens), i + batch)]
                nPast = min(self.nCtx - len(nBatch), len(self.inputIds[: self.pastTokens]))
                nTokens = len(nBatch)

                evalCode = llama_cpp.llama_eval(ctx=self.context,
                                                tokens=(llama_cpp.llama_token * len(nBatch))(*nBatch),
                                                n_tokens=nTokens,
                                                n_past=nPast,)

                if evalCode != 0:
                    self.warnAndExit("evaluate", f"Error occured during evaluation of {len(tokens)} tokens with batch"
                                                 f"size {batch}")

                self.inputIds[self.pastTokens : self.pastTokens + nTokens] = nBatch
                rows = nTokens if self.parameters.logitsAll else 1
                cols = self.nVocab
                offset = (0 if self.parameters.logitsAll else nTokens-1)
                self.scores[self.pastTokens+offset:self.pastTokens+nTokens, :].reshape(-1)[:] \
                    = llama_cpp.llama_get_logits(self.context)[: rows * cols]
                #print(f"pastTokens: {self.pastTokens} Tokens: {self.inputIds[self.pastTokens : self.pastTokens + nTokens]} String: {self.tokensToString(self.inputIds[self.pastTokens : self.pastTokens + nTokens])}") useful for debugging
                self.pastTokens += nTokens

    def sampleTokenWithModel(self):
        lastNTokensSize = self.parameters.n_keep if self.parameters.n_keep != -1 else self.nCtx
        lastNTokensData = [llama_cpp.llama_token(0)] * max(
            0, self.parameters.n_keep - len(self.inputIds[: self.pastTokens])
        ) + self.inputIds[: self.pastTokens][-lastNTokensSize :].tolist()

        lastNTokensData = (llama_cpp.llama_token * lastNTokensSize)(*lastNTokensData)

        logits: npt.NDArray[np.single] = self.scores[: self.pastTokens, :][-1, :]

        candidates = self.pCandidates
        candidatesData = self.pCandidatesData
        candidatesData["id"][:] = self.pCandidatesDataId
        candidatesData["logit"][:] = logits
        candidatesData["p"][:] = self.pCandidatesDataP
        candidates.data = candidatesData.ctypes.data_as(llama_cpp.llama_token_data_p)
        candidates.sorted = llama_cpp.c_bool(False)
        candidates.size = llama_cpp.c_size_t(self.nVocab)

        # actually sample the token
        llama_cpp.llama_sample_repetition_penalty(ctx=self.context,
                                                  candidates=llama_cpp.ctypes.byref(candidates),
                                                  last_tokens_data=lastNTokensData,
                                                  last_tokens_size=lastNTokensSize,
                                                  penalty=self.parameters.repeat_penalty)

        llama_cpp.llama_sample_frequency_and_presence_penalties(ctx=self.context,
                                                                candidates=llama_cpp.ctypes.byref(candidates),
                                                                last_tokens_data=lastNTokensData,
                                                                last_tokens_size=lastNTokensSize,
                                                                alpha_presence=self.parameters.presence_penalty,
                                                                alpha_frequency=self.parameters.frequency_penalty)

        if self.parameters.temperature == 0.0:
            id = llama_cpp.llama_sample_token_greedy(ctx=self.context,
                                                     candidates=llama_cpp.ctypes.byref(candidates))
        elif self.parameters.mirostat == 1:
            mirostatMU = llama_cpp.c_float(2.0*self.parameters.mirostat_tau)
            mirostatM = llama_cpp.c_int(100)
            llama_cpp.llama_sample_temperature(ctx=self.context,
                                               candidates=llama_cpp.ctypes.byref(candidates),
                                               temp=self.parameters.temperature)
            id = llama_cpp.llama_sample_token_mirostat(ctx=self.context,
                                                       candidates=llama_cpp.ctypes.byref(candidates),
                                                       tau=self.parameters.mirostat_tau,
                                                       eta=self.parameters.mirostat_eta,
                                                       mu=llama_cpp.ctypes.byref(mirostatMU),
                                                       m=mirostatM)
        elif self.parameters.mirostat == 2:
            mirostatMU = llama_cpp.c_float(2.0*self.parameters.mirostat_tau)
            llama_cpp.llama_sample_temperature(ctx=self.context,
                                               candidates=llama_cpp.ctypes.byref(candidates),
                                               temp=self.parameters.temperature)
            id = llama_cpp.llama_sample_token_mirostat_v2(ctx=self.context,
                                                          candidates=llama_cpp.ctypes.byref(candidates),
                                                          tau=self.parameters.mirostat_tau,
                                                          eta=self.parameters.mirostat_eta,
                                                          mu=llama_cpp.ctypes.byref(mirostatMU))
        else:   # temperature sampling
            llama_cpp.llama_sample_top_k(ctx=self.context,
                                         candidates=llama_cpp.ctypes.byref(candidates),
                                         k=self.parameters.top_k,
                                         min_keep=llama_cpp.c_size_t(1))
            llama_cpp.llama_sample_tail_free(ctx=self.context,
                                             candidates=llama_cpp.ctypes.byref(candidates),
                                             z=self.parameters.tfs_z,
                                             min_keep=llama_cpp.c_size_t(1))
            llama_cpp.llama_sample_typical(ctx=self.context,
                                           candidates=llama_cpp.ctypes.byref(candidates),
                                           p=llama_cpp.c_float(1.0),
                                           min_keep=llama_cpp.c_size_t(1))
            llama_cpp.llama_sample_top_p(ctx=self.context,
                                         candidates=llama_cpp.ctypes.byref(candidates),
                                         p=self.parameters.top_p,
                                         min_keep=llama_cpp.c_size_t(1))
            llama_cpp.llama_sample_temperature(ctx=self.context,
                                               candidates=llama_cpp.ctypes.byref(candidates),
                                               temp=self.parameters.temperature)
            id = llama_cpp.llama_sample_token(ctx=self.context,
                                              candidates=llama_cpp.ctypes.byref(candidates))
        return id

    def generateTokens(self, tokens: List[int]):
        nTokens = 0
        while True:
            self.evaluate(tokens=tokens, batch=32)
            newToken = self.sampleTokenWithModel()
            tokensON = yield newToken
            tokens = [newToken]
            #print(f"LLMCore: generateTokens: new token: {newToken}")
            if tokensON:
                tokens.extend(tokensON)

            nTokens += 1

    def tokenToByte(self, token: int) -> bytes:
        size = 32
        buffer = (llama_cpp.ctypes.c_char * size)()
        n = llama_cpp.llama_token_to_piece(self.model, llama_cpp.llama_token(token), buffer, size)

        assert n <= size
        return bytes(buffer[:n])  # no llama1 support

    def tokensToString(self, tokens: List[int]) -> str:
        buf = b""
        for token in tokens:
            buf += self.tokenToByte(token=token)
        return buf.decode("utf8", errors="ignore")

    def generate(self, stream: bool = False) -> str:   # streaming disabled for now
        antiPrompts: List[str] = self.parameters.antiPrompt
        tempBytes = b""
        finalString = ""
        tokens: List[int] = []
        tokenizedPromptTokens: List[int] = (self.tokenizeFull(self.parameters.prompt) if self.parameters.prompt != ""
                                            else [llama_cpp.llama_token_bos(self.context)])

        if len(tokenizedPromptTokens) >= self.parameters.nCtx:
            print(f"{tokenizedPromptTokens} tokens were requested to be processed, maximum is "
                  f"{llama_cpp.llama_n_ctx(self.context)}")
            return ""

        llama_cpp.llama_reset_timings(self.context)
        if antiPrompts != []:
            encodedAntiPrompts: List[bytes] = [a.encode("utf8") for a in antiPrompts]
        else:
            encodedAntiPrompts: List[bytes] = []

        incompleteFix: int = 0
        for t in self.generateTokens(tokens=tokenizedPromptTokens):  # should probably remove either tempbytes or
            if t == self.EOSToken:                                   # finalstring
                finalString = self.tokensToString(tokens=tokens) if len(finalString)+1 != len(tokens) else finalString
                #if len(tokens) <= 1:
                #    continue
                break
            tokens.append(t)
            #tokens.append(1)
            tempBytes += self.tokenToByte(token=tokens[-1])

            for k, char in enumerate(tempBytes[-3:]):
                k = 3 - k
                for number, pattern in [(2, 192), (3, 224), (4, 240)]:
                    if number > k and pattern & char == pattern:
                        print(str(number) + " " + str(pattern))
                        incompleteFix = number - k

            if incompleteFix > 0:
                incompleteFix -= 1
                continue

            antiPrompt = [a for a in encodedAntiPrompts if a in tempBytes]
            if len(antiPrompt) > 0:
                firstAntiPrompt = antiPrompt[0]
                tempBytes = tempBytes[: tempBytes.index(firstAntiPrompt)]
                break

            # implement streaming

            if len(tokens) > self.parameters.n_predict:
                finalString = self.tokensToString(tokens=tokens)
                break

        llama_cpp.llama_print_timings(self.context)
        return finalString if finalString != "" else tempBytes.decode("utf-8", errors="ignore")

    def response(self, stream: bool = False) -> str:    # streaming disabled for now
        if not stream:
            return self.generate(stream=stream)
        else:
            return "placeholder"

    def loadPrompt(self, path: str = None, prompt: str = None, type: str = None):
        supportedPromptTypes = ['alpaca', 'pygmalion', 'pygmalion2', 'zephyr-beta', 'openhermes-mistral']

        if path is not None:
            with open(path) as f:
                self.parameters.prompt = (" " + (f.read()).replace("{llmName}", self.parameters.modelName))
                self.parameters.prompt.replace("\\n", '\n')

            if type.lower() not in supportedPromptTypes:
                print(f"Prompt type not supported. Prompt type: {type.lower()}")
                self.global_go = False
                pass
        elif prompt is not None:
            self.parameters.prompt = prompt
        else:
            print("LLMCore: No prompt loaded.")

        self._promptTemplate = ""
        #TODO: fix prompt types, currently mistral has the sole working prompt style
        if type.lower() == "alpaca":
            self.systemPromptPrefix = ""
            self.inputPrefix = "### Instruction:"
            self.outputPrefix = "### Response:"
        elif type.lower() == "pygmalion":
            self.systemPromptPrefix = "{llmName}}'s Persona:"
            self.systemPromptSplitter = "<START>"
            self.inputPrefix = "You:"
            self.outputPrefix = ('[' + self.parameters.modelName + ']' + ':' + ' ')
            self.parameters.prompt.replace("PYGMALION", " ")
        elif type.lower() == "pygmalion2":
            self.systemPromptPrefix = ""
            self.inputPrefix = "<|user|>"
            self.outputPrefix = "<|model|>"
            self.parameters.prompt.replace("PYGMALION2", "")
        elif type.lower() == "zephyr-beta":
            self.systemPromptPrefix = "<|system|>"
            self.systemPromptSplitter = "</s>"
            self.userInputPrefix = "<|user|>"
            self.llmOutputPrefix = "<|assistant|>"
            self.inputSuffix = "</s>"
            self._promptTemplate = f"{self.userInputPrefix}\n[inputText]{self.inputSuffix}\n" \
                                   f"{self.llmOutputPrefix}\n"
        elif type.lower() == "openhermes-mistral":
            self.systemPromptSplitter = "<|im_end|>"
            self.systemPromptPrefix = "<|im_start|>system"
            self.inputPrefix = "<|im_start|>"
            self.inputSuffix = "<|im_end|>"
            self._promptTemplate = f"{self.inputPrefix}user\n[inputText]{self.inputSuffix}\n" \
                                  f"{self.inputPrefix}assistant\n"

    def promptTemplate(self, inputText: str = ""):
        return self._promptTemplate.replace("[inputText]", inputText)

    def manipulatePrompt(self, new, setting):
        pass    # add some functionality to mess with the prompt during runtime

    def printPrompt(self):
        if self.parameters.prompt:
            print(self.parameters.prompt)

    def exit(self):
        llama_cpp.llama_print_timings(self.context)
        llama_cpp.llama_free(self.context)

class LlamaOrig:
    def __init__(self, params):
        self.parameters = params
        self.llama = Llama(model_path=self.parameters.modelPath,
                           main_gpu=self.parameters.mainGPU,
                           n_gpu_layers=-1,
                           n_ctx=1024,
                           seed=int(randint(0, int(time()))),
                           n_threads=16)

    def response(self, stream=False): # placeholder argument
        res = str(self.llama(self.parameters.prompt, max_tokens=self.parameters.n_predict,
                              mirostat_mode=2,
                              presence_penalty=self.parameters.presence_penalty,
                              frequency_penalty=self.parameters.frequency_penalty,
                              mirostat_eta=self.parameters.mirostat_eta,
                              mirostat_tau=self.parameters.mirostat_tau,
                              repeat_penalty=self.parameters.repeat_penalty,
                              temperature=self.parameters.temperature,
                              top_k=self.parameters.top_k,
                              top_p=self.parameters.top_p,
                              stop=self.parameters.antiPrompt))

        textOutputStart = res.find("'text':") + 9
        textOutputEnd = res.find("index") - 4
        textOutput = res[textOutputStart:textOutputEnd]
        print(textOutput)
        return textOutput

    def loadPrompt(self, path: str = None, prompt: str = None, type: str = None):
        supportedPromptTypes = ['alpaca', 'pygmalion', 'pygmalion2', 'zephyr', 'openhermes-mistral']

        if path is not None:
            with open(path) as f:
                self.parameters.prompt = (" " + (f.read()).replace("{llmName}", self.parameters.modelName))
                self.parameters.prompt.replace("\\n", '\n')

            if type.lower() not in supportedPromptTypes:
                print(f"Prompt type not supported. Prompt type: {type.lower()}")
                self.global_go = False
                pass
        elif prompt is not None:
            self.parameters.prompt = prompt
        else:
            print("LLMCore: No prompt loaded.")

        #TODO: fix prompt types, currently mistral has the sole working prompt style
        if type.lower() == "alpaca":
            self.systemPromptPrefix = ""
            self.inputPrefix = "### Instruction:"
            self.outputPrefix = "### Response:"
        elif type.lower() == "pygmalion":
            self.systemPromptPrefix = "{llmName}}'s Persona:"
            self.systemPromptSplitter = "<START>"
            self.inputPrefix = "You:"
            self.outputPrefix = ('[' + self.parameters.modelName + ']' + ':' + ' ')
            self.parameters.prompt.replace("PYGMALION", " ")
        elif type.lower() == "pygmalion2":
            self.systemPromptPrefix = ""
            self.inputPrefix = "<|user|>"
            self.outputPrefix = "<|model|>"
            self.parameters.prompt.replace("PYGMALION2", "")
        elif type.lower() == "zephyr":
            self.systemPromptPrefix = ""
            self.inputPrefix = "</s><|user|>"
            self.outputPrefix = "</s><|assistant|>"
        elif type.lower() == "openhermes-mistral":
            self.systemPromptSplitter = "<|im_end|>"
            self.systemPromptPrefix = "<|im_start|>system"
            self.inputPrefix = "<|im_start|>"
            self.inputPostfix = "<|im_end|>"

    def promptTemplate(self):
        template = f"{self.inputPrefix}[inputName]:\n[inputText]{self.inputPostfix}\n"
        template += f"{self.inputPrefix}[outputName]:\n"
        return template

    def reset(self):
        pass # placeholder
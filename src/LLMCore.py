import json

import llama_cpp
from llama_cpp import Llama
import LLMUtils
import numpy as np
import multiprocessing as mp
from time import time
from random import randint

# Explanation for these parameters is found at https://abetlen.github.io/llama-cpp-python/

def defaultLlamactxParams():
    return LLMUtils.LLMConfig()

class LlamaModel:
    def __disabledinit__(self, parameters):
        self.parameters = parameters
        self.ctxParams = self.parameters.getCtxParms()
        self.modelParams = self.parameters.getModelParams()
        self.modelPath = self.parameters.modelPath
        self.ctxParams.n_ctx = 1024
        self.ctxParams.n_gpu_layers = self.parameters.nOffloadLayer
        self.ctxParams.main_gpu = self.parameters.mainGPU

        self.threadCount = mp.cpu_count()

        if self.ctxParams.seed <= 0:
            self.ctxParams.seed = int(randint(0, int(time())))

        if self.threadCount < 6:
            print("Low thread count. Inference might be slow.")

        if self.modelPath != None:
            try:
                self.model = llama_cpp.llama_load_model_from_file(self.modelPath.encode('utf-8'), self.modelParams)
                self.context = llama_cpp.llama_new_context_with_model(self.model, self.ctxParams)
            except FileExistsError:
                print("Invalid filepath")

        else:
            raise FileExistsError("No model path specified.")

        self.n_ctx = llama_cpp.llama_n_ctx(self.context)
        self.remainingTokens = self.parameters.n_predict

        self._init()
        self.antiPromptTokens()

    def __init__(self, parameters):
        self.parameters = parameters
        self.llama = Llama(model_path=parameters.modelPath,
                           main_gpu=parameters.mainGPU,
                           seed=int(randint(0, int(time()))),
                           n_threads=16)
        self.loadPrompt(path=None, prompt="", type=self.parameters.modelType)

    def tempGenerate(self):
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

    def update(self, parameters):
        self.parameters = parameters
        self.n_ctx = llama_cpp.llama_n_ctx(self.context)
        self.remainingTokens = self.parameters.n_predict

    def loadPrompt(self, path=None, prompt=None, type=None):
        supportedPromptTypes = ['alpaca\n', 'pygmalion\n', 'pygmalion2\n', 'zephyr\n', 'openhermes-mistral\n']
        promptType = type+'\n'
        self.parameters.prompt = prompt

        if path is not None:
            with open(path) as f:
                self.parameters.prompt = (" " + (f.read()).replace("{llmName}", self.parameters.modelName))
                self.parameters.prompt.replace("\\n", '\n')

            with open(path) as f:
                promptType = f.readline()

            if promptType.lower() not in supportedPromptTypes:
                print(f"Prompt type not supported. Prompt type: {promptType.lower()}")
                self.global_go = False
                pass

        #TODO: expand and fix systemprompt prefixes
        if promptType.lower() == "alpaca\n":
            self.systemPromptPrefix = ""
            self.inputPrefix = "### Instruction:\n"
            self.outputPrefix = "### Response:\n"
        elif promptType.lower() == "pygmalion\n":
            self.systemPromptPrefix = "{llmName}}'s Persona:"
            self.systemPromptSplitter = "<START>"
            self.inputPrefix = "You:"
            self.outputPrefix = ('[' + self.parameters.modelName + ']' + ':' + ' ')
            self.parameters.prompt.replace("PYGMALION", " ")
        elif promptType.lower() == "pygmalion2\n":
            self.systemPromptPrefix = ""
            self.inputPrefix = "<|user|> "
            self.outputPrefix = "<|model|> "
            self.parameters.prompt.replace("PYGMALION2", "")
        elif promptType.lower() == "zephyr\n":
            self.systemPromptPrefix = ""
            self.inputPrefix = "</s>\n<|user|>"
            self.outputPrefix = "</s>\n<|assistant|>"
        elif promptType.lower() == "openhermes-mistral\n":
            self.systemPromptSplitter = ""
            self.systemPromptPrefix = "<|im_start|>system"
            self.inputPrefix = "<|im_start|>"
            self.outputPrefix = "<|im_end|>"

    def manipulatePrompt(self, new, setting):
        print(type(self.parameters.prompt))

        if setting == -1:
            self.parameters.prompt += new
        elif setting == 0:
            self.parameters.prompt += self.outputPrefix + ' ' + new + '\n' + self.inputPrefix
        elif setting == 1:
            self.parameters.prompt += self.inputPrefix + ' ' + new + '\n' + self.outputPrefix
        elif setting == 2:
            self.parameters.prompt = new
        else:
            raise RuntimeError(f"Wrong setting ({setting}) supplied.")

    def printPrompt(self):
        if self.parameters.prompt:
            print(self.parameters.prompt)

    def tokenize(self, input, bos=True):
        input = input.encode('utf-8')
        embedding_input = (llama_cpp.llama_token * ((len(input) + 1) * 4))()
        nTokens = llama_cpp.llama_tokenize(self.model, bytes(input), len(input), embedding_input, self.ctxParams.n_ctx, bos)  # bytes could potentially not work, replace with utf8 encoding
        return embedding_input[:nTokens]

    def tokenizePrompt(self):
        self.embedding_input = self.tokenize(self.parameters.prompt)

        # check prompt length
        if len(self.embedding_input) > (self.n_ctx - 4):
            raise RuntimeError(
                f"Prompt size is too long, prompt loaded is {len(self.embedding_input)} tokens however the maximum is {self.n_ctx - 4}")

    def _init(self):
        self.loadPrompt(path=None, prompt="", type=self.parameters.modelType)   # this should not be handled i here
        self.t_inputPrefix = self.tokenize(self.inputPrefix)                    # rewrite at some point
        self.t_outputPrefix = self.tokenize(self.outputPrefix, False)
        self.newlineToken = self.tokenize(self.parameters.newlineToken, False)
        self.EOSToken = self.tokenize(self.parameters.EOT)
        # self.antiPrompt = self.tokenize(self.ctxParams.antiprompt)

        #print(f"Prompt has {len(self.embedding_input)} tokens")

    def antiPromptTokens(self):
        self.antiPrompt = []

        for i in self.parameters.antiPrompt:
            self.antiPrompt.append(self.tokenize(i, False))

    def generate(self):
        self.tokenizePrompt()

        if self.parameters.n_keep < 0 or self.parameters.n_keep > len(self.embedding_input):
            self.parameters.n_keep = len(self.embedding_input)

        self.remainingTokens = self.parameters.n_predict
        self.lastNTokens = [0]*self.n_ctx
        self.inputConsumed = 0
        self.embedding = []
        self.n_past = 0
        self.length = 0

        print(f"Prompt has {len(self.embedding_input)} tokens")

        # actual output evaluation
        while self.remainingTokens > 0 or self.parameters.n_predict == -1:  # -1 for infinite generation
            print("test123 1 " + str(len(self.embedding_input)))
            if len(self.embedding) > 0:
                print("test123 2")
                if (self.n_past + len(self.embedding) > self.n_ctx):
                    n_left = self.n_past - self.parameters.n_keep
                    self.n_past = self.parameters.n_keep

                    insert = self.lastNTokens[self.n_ctx - int(n_left/2) - len(self.embedding):-len(self.embedding)]
                    self.embedding = insert + self.embedding

                if llama_cpp.llama_eval(self.context, (llama_cpp.llama_token * len(self.embedding))(*self.embedding), len(self.embedding), self.n_past) != 0:
                    raise Exception("Failed at llama_ecal")

            self.n_past += len(self.embedding)
            self.embedding = []

            if len(self.embedding_input) <= self.inputConsumed:
                print("test123 3")
                topk = llama_cpp.llama_n_vocab(self.context) if self.parameters.top_k <= 0 else self.parameters.top_k
                repeatLastN = self.n_ctx if self.parameters.repeat_last_n < 0 else self.parameters.repeat_last_n

                logits = llama_cpp.llama_get_logits(self.context)
                nVocab = llama_cpp.llama_n_vocab(self.model)

                # for k, v in self.parameters.logit_bias.items(): # TODO: enable manually applying a bias to words at some point
                #     logits[k] += v
                print("Test" + str(nVocab))
                self.candidatesData = np.array(
                    [],
                        dtype=np.dtype(
                            [("id", np.intc), ("logit", np.single), ("p", np.single)]
                            ),
                        )
                self.candidatesData.resize(nVocab, refcheck=False)
                self.candidates = llama_cpp.llama_token_data_array(
                    data=self.candidatesData.ctypes.data_as(llama_cpp.llama_token_data_p),
                    size=nVocab,
                    sorted=False
                )

                pCandidates = self.candidates
                pCandidates.data = self.candidatesData.ctypes.data_as(llama_cpp.llama_token_data_p)
                pCandidates.sorted = llama_cpp.c_bool(False)
                pCandidates.size = llama_cpp.c_size_t(nVocab)

                nlLogit = logits[llama_cpp.llama_token_nl(self.context)] # fix that
                lastNRepeat = min(len(self.lastNTokens), repeatLastN, self.n_ctx)

                self.candidatesData = (llama_cpp.llama_token * lastNRepeat)(*self.lastNTokens[len(self.lastNTokens) - lastNRepeat:])

                llama_cpp.llama_sample_repetition_penalty(self.context, llama_cpp.ctypes.byref(pCandidates), self.candidatesData, lastNRepeat, llama_cpp.c_float(self.parameters.repeat_penalty))
                llama_cpp.llama_sample_frequency_and_presence_penalties(self.context, pCandidates, self.candidatesData, lastNRepeat, llama_cpp.c_float(self.parameters.frequency_penalty), llama_cpp.c_float(self.parameters.presence_penalty))

                if not self.parameters.penalize_nl:
                       logits[llama_cpp.llama_token_nl(self.context)] = nlLogit # fix that as well

                if self.parameters.temperature <= 0:
                    id = llama_cpp.llama_sample_token_greedy(self.context, pCandidates)
                else:
                    if self.parameters.mirostat == 1:
                        mirostat_mu = 2.0 * self.parameters.mirostat_tau
                        mirostat_m = 100

                        llama_cpp.llama_sample_temperature(self.context, pCandidates, llama_cpp.c_float(self.parameters.temperature))
                        id = llama_cpp.llama_sample_token_mirostat(self.context, pCandidates,
                                                                   llama_cpp.c_float(self.parameters.mirostat_tau),
                                                                   llama_cpp.c_float(self.parameters.mirostat_eta),
                                                                   llama_cpp.c_int(mirostat_m), llama_cpp.c_float(mirostat_mu))
                    elif self.parameters.mirostat == 2:
                        mirostat_mu = 2.0 * self.parameters.mirostat_tau

                        llama_cpp.llama_sample_temperature(self.context, pCandidates, llama_cpp.c_float(self.parameters.temperature))
                        id = llama_cpp.llama_sample_token_mirostat_v2(self.context, pCandidates,
                                                                   llama_cpp.c_float(self.parameters.mirostat_tau),
                                                                   llama_cpp.c_float(self.parameters.mirostat_eta),
                                                                   llama_cpp.c_float(mirostat_mu))
                    else:
                        llama_cpp.llama_sample_top_k(self.context, pCandidates, topk, min_keep=llama_cpp.c_size_t(1))
                        llama_cpp.llama_sample_tail_free(self.context, pCandidates, llama_cpp.c_float(self.parameters.tfs_z), min_keep=llama_cpp.c_size_t(1))
                        llama_cpp.llama_sample_typical(self.context, pCandidates, llama_cpp.c_float(self.parameters.typical_p), min_keep=llama_cpp.c_size_t(1))
                        llama_cpp.llama_sample_top_p(self.context, pCandidates, llama_cpp.c_float(self.parameters.top_p), min_keep=llama_cpp.c_size_t(1))
                        llama_cpp.llama_sample_temperature(self.context, pCandidates, llama_cpp.c_float(self.parameters.temperature))
                        id = llama_cpp.llama_sample_token(self.context, pCandidates)

                self.lastNTokens.pop(0)
                self.lastNTokens.append(id)

                if(id == llama_cpp.llama_token_eos(self.context)):
                    return None
                    #print(f"EOS: {llama_cpp.llama_token_eos(self.context)}")
                    #id = self.newlineToken[0]
                    #self.embedding.append(id)
                    #if(len(self.antiPrompt) > 0):
                        #self.embedding_input += self.antiPrompt[0]
                        #for id in self.antiPrompt[0]:
                            #self.embedding.append(id)
                else:
                    self.embedding.append(id)
                    self.length += 1

                self.remainingTokens -= 1

            else:
                while len(self.embedding_input) > self.inputConsumed:
                    print("test123 4 " + str(self.inputConsumed))
                    self.embedding.append(self.embedding_input[self.inputConsumed])
                    self.lastNTokens.pop(0)
                    self.lastNTokens.append(self.embedding_input[self.inputConsumed])
                    self.inputConsumed += 1
                    # add batch functionality
                    if len(self.embedding) >= self.parameters.n_batch:
                        break


            for id in self.embedding[:self.length]:
                yield id

            if len(self.embedding) > 0 and self.embedding[-1] == llama_cpp.llama_token_eos(self.context):
                for i in self.EOSToken:
                    yield i
                break

    def output(self):
        self.extraCharFix = []

        for i in self.generate():
            char = self.tokenToString(i)

            if None in self.extraCharFix:
                self.extraCharFix[self.extraCharFix.index(None)] = char

            if len(self.extraCharFix) > 0 and not None in self.extraCharFix:
                #self.parameters.prompt += (b"".join(self.extraCharFix)).decode("utf8") # delete
                yield(b"".join(self.extraCharFix)).decode("utf8")

            for number, pattern in [(2, 192), (3, 224), (4, 240)]:
                if pattern & int.from_bytes(char, 'little') == pattern:
                    self.extraCharFix = [char] + ([None] * (number-1))

            if len(self.extraCharFix) > 0:
                continue

            #self.parameters.prompt += char.decode("utf8") # delete
            yield char.decode("utf8")

    def response(self, stream=False):
        if stream is False:
            temp = ""
            for i in self.output():
                temp += i
            return temp
        else:
            return self.output()

    def tokenToString(self, tokenid) -> bytes:
        size = 32
        buffer = (llama_cpp.ctypes.c_char * size)()
        n = llama_cpp.llama_token_to_piece(self.model, llama_cpp.llama_token(tokenid), buffer, size)
        assert n <= size
        return bytes(buffer[:n])

    def exit(self):
        llama_cpp.llama_print_timings(self.context)
        llama_cpp.llama_free(self.context)
import LLMCore
import LLMUtils
from LLMCore import LlamaModel
from LLMUtils import LLMConfig, defaultLlamactxParams
from memoryDB import MemoryDB
from dataclasses import dataclass
from datetime import datetime
from nrclex import NRCLex
from typing import List
import time
import math
import json


@dataclass
class StrIntPair:
    string: str
    integer: int


class Melba:
    def __init__(self, modelPath: str, databasepath: str, llmconfig: LLMConfig = None, logPath: str = None):
        self.llmConfig = self.defaultConfig() if llmconfig is None else llmconfig
        self.llmConfig.modelPath = modelPath
        self.stage = 0

        self.llm = LlamaModel(self.llmConfig)
        self.memory = Memory(databasePath=databasepath, logPath=logPath)
        self.context = Context(memoryDB=self.memory)
        self.utils = MelbaTools(memoryDB=self.memory)
        self.logger = Logger(logPath)
        self.emotionHandler = EmotionHandler()

        self.llm.loadPrompt(path=None, prompt="", type=self.llmConfig.modelType)

    def defaultConfig(self):
        llmConfig = defaultLlamactxParams()
        llmConfig.nCtx = 1024
        llmConfig.n_keep = 1024
        llmConfig.n_predict = 512
        llmConfig.modelName = "Melba"
        llmConfig.modelType = "openhermes-mistral"
        llmConfig.antiPrompt = ["<"]
        llmConfig.mirostat = 2
        llmConfig.mirostat_tau = 3.0
        llmConfig.mirostat_eta = 0.25
        llmConfig.frequency_penalty = 0.4
        llmConfig.repeat_penalty = 1.2
        llmConfig.top_p = 0.65
        llmConfig.top_k = 30
        llmConfig.temperature = 0.65
        llmConfig.logit_bias = {32000 : 1.1}
        llmConfig.nOffloadLayer = 100
        llmConfig.mainGPU = 0

        return llmConfig

    def getCurrentConfig(self):
        return self.llmConfig

    # TODO: update this function
    def updateLLMConfig(self, newConfig):
        newconfig = self.getCurrentConfig()
        attributes = ["n_keep", "n_predict", "tfs_z", "typical_p",
                      "top_k", "top_p", "temperature", "mirostat",
                      "mirostat_tau", "mirostat_eta", "repeat_last_n", "repeat_penalty",
                      "frequency_penalty", "presence_penalty", "penalize_nl", "n_batch"]

        for attribute in attributes:
            setattr(newconfig, attribute, json.loads(newConfig)[attribute])

        self.llm.update(newconfig)

    def setStage(self, stage: int = 0):
        self.stage = stage

    def setVision(self, setting: bool,
                        parameters: LLMConfig = None, encoderPath = None, systemPrompt = None, userPrompt = None):
        if setting and not self.context.vllm:
            self.context.vllm = VisionHandler(parameters=parameters,
                                              encoderPath=encoderPath,
                                              systemPrompt=systemPrompt, userPrompt=userPrompt)
        if not setting and self.context.vllm:
            self.context.vllm.unload()

    def prompt(self, person: str, message: str) -> str:
        message = self.utils.preprocessMessage(message=message)
        if message == "":
            return ""

        systemprompt = self.memory.systemPrompt()
        personality = self.memory.personality()
        context = self.context.situationalContext(person=person, message=message)
        pastconversation = self.memory.savedChat(username=person)

        self.convoStyle = self.llm.promptTemplate(inputText=message)

        finalPrompt = (f"{self.llm.systemPromptPrefix} {systemprompt.replace('[personality]', personality)}\n"
                       f"{self.llm.systemPromptSplitter}\n"
                       f"{pastconversation}\n" + self.convoStyle
                       ).replace("[context]", context).replace("{llmName}", self.llmConfig.modelName)

        return finalPrompt

    def getMelbaResponse(self, message: str, person: str) -> str:
        self.curPrompt = self.prompt(person=person, message=message)
        if self.curPrompt == "" or message[0] == '@':
            self.logger.log(
                        message="melbaToast: Something went wrong while constructing the prompt, please restart Melba."
            )
            return json.dumps({'response': "", 'emotions' : ['neutral']})
        self.llm.loadPrompt(path=None, prompt=self.curPrompt, type=self.llmConfig.modelType)

        self.logger.log(message=f"melbaToast: Current prompt is: '{self.curPrompt}'\n")

        response = self.llm.response(stream=False)
        self.emotionHandler.evaluateEmotion(text=response)
        emotion = self.emotionHandler.getEmotion()
        filteredResponse = self.utils.filterMessage(message=response)

        self.memory.saveConversation(person=person,
                                     conversation=f"{self.memory.savedChat(username=person)}\n"
                                                  f"{self.convoStyle + response + self.llm.inputSuffix}")
        self.llm.reset()  # needs to be improved
        self.logger.log(message=f"User: '{person}'\tMessage: '{message}'\tMelba response: '{response}'"
                                f"\tEmotions: "f"'{emotion}'")

        return json.dumps({'response' : filteredResponse, 'emotions' : emotion})

    def regenerateResponse(self, stream=False):
        if stream:
            for token in self.llm.response():
                yield token
        else:
            return self.llm.response()


class Memory:
    def __init__(self, databasePath: str, logPath: str = None):
        self.memoryDB = MemoryDB(databasePath)
        self.logger = Logger(logPath)

    def accessMemories(self, type: str, identifier: str, vectorSearch: bool = False) -> str:
        response = self.memoryDB.metadataQueryDB(type=type, identifier=identifier) if not vectorSearch else\
                   json.loads(
                       json.dumps(self.memoryDB.vectorQueryDB(queries=[identifier], filter=type, nResults=1))
                   )['documents']

        if response == "":
            self.logger.log(message=f"melbaToast: No '{type}' memory with identifier '{identifier}' found.")

        return response

    def savedChat(self, username: str):
        return self.accessMemories(type="savedChat", identifier=username)

    def systemPrompt(self):
        response = self.accessMemories(type="systemPrompt", identifier="Melba Toast", vectorSearch=True)[0][0]
        return self.accessMemories(type="systemPrompt", identifier="generic") if response == "" else response

    def personality(self):
        return self.accessMemories(type="personality", identifier="generic2")

    def personalInformation(self, name: str):
        return self.accessMemories(type="personalinformation", identifier=name)

    def updateMemory(self, type: str, identifier: str, newContent: str):
        self.memoryDB.updateOrCreateDBEntry(type=type, identifier=identifier, content=newContent)
        self.logger.log(message=f"melbaToast: Updated memory entry of type '{type}' "
                                f"with identifier '{identifier}' and content '{newContent}'")

    def saveConversation(self, person: str, conversation: str):
        lines = conversation.count("<|im_start|>")  # TODO: change to more accurate way of measuring message count
        if lines > 2:
            lines = '\n'.join(conversation.split('\n')[4:])
        else:
            lines = conversation
        self.updateMemory(type="savedChat", identifier=person, newContent=f"{lines}")

    def wipeDB(self):
        self.memoryDB.chromaClient.reset()
        self.logger.log(message="melbaToast: MemoryDB fully wiped.")


class Context:
    def __init__(self, memoryDB: Memory):
        self.memoryDB = memoryDB.memoryDB
        self.vllm = None
        self.image = None

    # TODO: increase search accuracy and response volume (reimplementation)
    def returnWebContent(self, searchQuery: str = None) -> List[str]:
        return ["placeholder"]

    def setImage(self, image: str):
        if image is None:
            self.image = None
            pass
        self.image = image

    def situationalContext(self, person: str, message: str) -> str:
        vectorstorageresponse = self.memoryDB.vectorQueryDB(queries=[message], filter="information", nResults=2)
        context = []
        print(vectorstorageresponse)

        index = 0
        print(f"Len: {len(vectorstorageresponse['ids'][0])}")
        for i in range(len(vectorstorageresponse['ids'][0])):
            if vectorstorageresponse['distances'][0][index] > 0.5:
                print(f"distance: {vectorstorageresponse['distances'][0][index]}")
                context.append(vectorstorageresponse['documents'][0][index])
            index += 1
        formattedcontext = ""
        for c in context:
            formattedcontext += f"{c}\n"
        formattedcontext += f"Current date: {datetime.today().strftime('%d/%m/%Y')}\n" \
                            f"Current time: {datetime.now().strftime('%H:%M')}\n" \
                            f"You are speaking to {person}"
        if self.vllm and self.image:
            formattedcontext += f"\nA description of what you are currently viewing: " \
                                f"'{self.vllm.visualDescription(imageurl=self.image)}'"
        print(f"melbaToast: Formatted context: {formattedcontext}")
        return formattedcontext


class VisionHandler:
    def __init__(self, parameters: LLMUtils.LLMConfig, encoderPath: str,
                       systemPrompt: str, userPrompt: str):
        llavamodel = LLMCore.LlamaModel(parameters=parameters if parameters is not None else self.defaultVisionConfig())
        self.llavamodel = LLMCore.LlamaLlavaModel(llamaModel=llavamodel,
                                                  modelPath=encoderPath,
                                                  parameters=parameters)
        self.sysprompt = systemPrompt
        self.userprompt = userPrompt

    def defaultVisionConfig(self) -> LLMConfig:
        llmConfig = defaultLlamactxParams()
        llmConfig.nCtx = 1024
        llmConfig.n_keep = 1024
        llmConfig.n_predict = 512
        llmConfig.modelName = "Melba"
        llmConfig.modelType = "openhermes-mistral"
        llmConfig.antiPrompt = ["<"]
        llmConfig.mirostat = 1
        llmConfig.mirostat_tau = 5.0
        llmConfig.mirostat_eta = 0.25
        llmConfig.mirostat_mu = 2.0
        llmConfig.frequency_penalty = 0.3
        llmConfig.repeat_penalty = 1.2
        llmConfig.top_p = 0.30
        llmConfig.top_k = 20
        llmConfig.temperature = 0.1
        llmConfig.logit_bias = {2: 0.95} # TODO: token shouldn't be hardcoded, change that
        llmConfig.nOffloadLayer = 100
        llmConfig.mainGPU = 0
        return llmConfig

    def visualDescription(self, imageurl: str) -> str:
        response = self.llavamodel.response(systemprompt=self.sysprompt,
                                            prompt=self.userprompt,
                                            imageurl=imageurl)
        print(f"Vision response: {response} - {imageurl}")
        return response

    def unload(self):
        self.llavamodel = None

class EmotionHandler:
    def __init__(self):
        self.currentEmotion = None

    def evaluateEmotion(self, text):
        n = NRCLex(text=text)
        currentemotions = []

        for emotion in n.top_emotions:
            currentemotions.append(emotion[0])

        self.currentEmotions = currentemotions

    def getEmotion(self):
        return self.currentEmotion


class MelbaTools:
    def __init__(self, memoryDB: Memory):
        self.memoryDB = memoryDB.memoryDB
        self.swearWords = ""  # TODO: list instead of string
        self.maliciousWords = ""  # TODO: List instead of string

    def characterFrequency(self, sentence: str) -> List[StrIntPair]:
        charfreq: List[StrIntPair] = []

        for c in sentence:
            exists = False
            for pair in charfreq:
                if pair.string == c:
                    pair.integer += 1
                    exists = True
            if not exists:
                pair = StrIntPair(c, 1)
                charfreq.append(pair)

        return charfreq

    def wordFrequency(self, sentence: str) -> List[StrIntPair]:
        wordfreq: List[StrIntPair] = []

        for word in sentence.split(' '):
            exists = False

            for pair in wordfreq:
                if pair.string == word:
                    pair.integer += 1
                    exists = True
            if not exists:
                pair = StrIntPair(word, 1)
                wordfreq.append(pair)

        return wordfreq

    def characterProbability(self, frequencies: List[StrIntPair], target: str):  # this function should likely be
        targetindex = 0                                                          # swapped out for something more
        wordsum = 0                                                              # accurate and performant

        iterator = 0
        for field in frequencies:
            wordsum += field.integer
            targetindex = iterator if field.string == target else targetindex
            iterator += 1

        return frequencies[targetindex].integer/wordsum

    def sentenceEntropy(self, sentence: str):
        frequencies: List[StrIntPair] = self.wordFrequency(sentence=sentence)
        entropy = 0.0

        for field in frequencies:
            charprob = self.characterProbability(frequencies=frequencies, target=field.string)
            entropy += charprob * math.log2(charprob)

        return -entropy

    def isSwearWord(self, word: str) -> bool:
        if self.swearWords == "":
            swearWords = (self.memoryDB.metadataQueryDB(type="swearwords", identifier="all")).split()
            self.swearWords = (word.replace(' ', '') for word in swearWords)
        if word.replace('\n', '').replace(' ', '') in self.swearWords:
            return True
        return False

    def maliciousWordsCount(self, words: List[str]) -> int:
        if self.maliciousWords == "":
            self.maliciousWords = self.memoryDB.metadataQueryDB(type="maliciouswords", identifier="all").split()

        mwordcount = 0
        for word in words:
            if word in self.maliciousWords:
                mwordcount += 1

        return mwordcount

    def filterMessage(self, message: str) -> str:
        filteredmessage = []

        for word in message.split():
            print(word)
            if self.isSwearWord(word=word):
                filteredmessage.append("[TOASTED]")
            else:
                filteredmessage.append(word)

        return ' '.join(filteredmessage)

    def preprocessMessage(self, message: str) -> str:
        if self.sentenceEntropy(sentence=message) > 1.5 and \
                self.maliciousWordsCount(words=message.split()) <= 0:
            print(self.sentenceEntropy(sentence=message))
            return message
        return ""
        # stage 1 will include a llm preprocessing this message and further deciding whether it is valid for
        # further use
        # stage 0 will use this function, though a better filter will be needed


class Logger:
    def __init__(self, logpath: str = None):
        if logpath is None:
            print("Logger: No logpath specified, logging disabled.")
        else:
            self.logPath = logpath

    def log(self, message: str):
        if self.logPath is not None:
            try:
                with open(self.logPath, mode="a") as file:
                    file.write(f"\n[{datetime.utcnow().strftime('%y-%m-d %H:%M:%S')}]-[{message}]")
                    print(message)
            except:
                FileExistsError("File does not exist.")


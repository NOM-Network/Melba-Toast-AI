from LLMCore import LlamaModel
from LLMUtils import LLMConfig, defaultLlamactxParams
from memoryDB import MemoryDB
from dataclasses import dataclass
from datetime import datetime
from nrclex import NRCLex
from typing import List
import math
import json


@dataclass
class StrIntPair:
    string: str
    integer: int


class Melba:
    def __init__(self, modelPath: str, databasepath: str, llmconfig: LLMConfig = None, logPath: str = None):
        self.defaultConfig() if llmconfig is None else llmconfig
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
        self.llmConfig = defaultLlamactxParams()
        self.llmConfig.nCtx = 1024
        self.llmConfig.n_keep = 1024
        self.llmConfig.n_predict = 128
        self.llmConfig.modelName = "Melba"
        self.llmConfig.modelType = "insert supported model"
        self.llmConfig.antiPrompt = ["<"]
        self.llmConfig.mirostat = 2
        self.llmConfig.mirostat_tau = 5.0
        self.llmConfig.mirostat_eta = 0.25
        self.llmConfig.frequency_penalty = 0.4
        self.llmConfig.repeat_penalty = 1.2
        self.llmConfig.top_p = 0.65
        self.llmConfig.top_k = 30
        self.llmConfig.temperature = 0.65
        self.llmConfig.logit_bias = {32000 : 1.5}
        self.llmConfig.nOffloadLayer = 100
        self.llmConfig.mainGPU = 0

    def getCurrentConfig(self):
        return self.llmConfig

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
    
    def prompt(self, person: str, message: str) -> str:
        message = self.utils.preprocessMessage(message=message)
        if message == "":
            return ""
        systemprompt = self.memory.systemPrompt()
        personality = self.memory.personality()

        # TODO: Implement retrieval augmented generation aka get relevant information into the context
        context = self.context.situationalContext(person=person, message=message)
        pastconversation = self.memory.savedChat(username=person)
        self.convoStyle = self.llm.promptTemplate(inputText=message)
        finalPrompt = (f"{self.llm.systemPromptPrefix} {systemprompt.replace('[personality]', personality)}\n"
                       f"{context}"
                       f"{self.llm.systemPromptSplitter}\n"
                       f"{pastconversation}\n" + self.convoStyle)

        finalPrompt = finalPrompt.replace("{llmName}", self.llmConfig.modelName)
        return finalPrompt

    def getMelbaResponse(self, message, person, stream=False) -> str:
        self.curPrompt = self.prompt(person=person, message=message)        # insert model specific tokens
        if self.curPrompt == "":
            return json.dumps({'response': "", 'emotions' : ['neutral']})
        self.llm.loadPrompt(path=None, prompt=self.curPrompt, type=self.llmConfig.modelType)

        if self.curPrompt == "":  # we shouldn't even get here
            self.logger.log(message="melbaToast: Something went wrong while constructing the prompt, please restart Melba.")
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
        lines = conversation.count("<|im")  # TODO: change to more accurate way of measuring message count
        if lines > 8:
            lines = '\n'.join(conversation.split('\n')[5:])
        else:
            lines = conversation
        self.updateMemory(type="savedChat", identifier=person, newContent=f"{lines}")

    def wipeDB(self):
        self.memoryDB.chromaClient.reset()
        self.logger.log(message="melbaToast: MemoryDB fully wiped.")


class Context:
    def __init__(self, memoryDB: Memory):
        self.memoryDB = memoryDB.memoryDB

    # TODO: increase search accuracy and response volume (reimplementation)
    def returnWebContent(self, searchQuery: str = None) -> List[str]:
        return ["placeholder"]

    def situationalContext(self, person: str, message: str) -> str:
        # stage 2(?) should enable a llm summarizing and formatting the message into a useful query for the vectordb
        vectorstorageresponse = self.memoryDB.vectorQueryDB(queries=[message], filter=None, nResults=2)
        # stage 2 should summarize the results into something usable           - needs to be filtered
        if vectorstorageresponse == "":
            webResponse = self.returnWebContent(searchQuery=message) # searchQuery will be extracted keywords
        return ""


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
        frequencies: List[StrIntPair] = self.characterFrequency(sentence=sentence)
        entropy = 0.0

        for field in frequencies:
            charprob = self.characterProbability(frequencies=frequencies, target=field.string)
            entropy += charprob * math.log2(charprob)

        return -entropy

    def isSwearWord(self, word: str) -> bool:
        if self.swearWords == "":
            self.swearWords = (self.memoryDB.metadataQueryDB(type="swearwords", identifier="all")).split()
        if word.replace('\n', '') in self.swearWords:
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
        if self.sentenceEntropy(sentence=message) > 3.0 and \
                self.maliciousWordsCount(words=message.split()) <= 0:
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

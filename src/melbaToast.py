from LLMCore import LlamaModel
from LLMUtils import LLMConfig, defaultLlamactxParams
from memoryDB import MemoryDB
from dataclasses import dataclass
from datetime import datetime
from nrclex import NRCLex
from typing import List
import requests
import math
import json
import yake


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
        self.logger = Logger(logPath)
        self.memory = Memory(databasePath=databasepath, logPath=logPath)
        self.emotionHandler = EmotionHandler()

    def defaultConfig(self):
        self.llmConfig = defaultLlamactxParams()
        self.llmConfig.nCtx = 1024
        self.llmConfig.n_keep = 1024
        self.llmConfig.modelName = "Melba"
        self.llmConfig.modelType = "openhermes-mistral"
        self.llmConfig.antiPrompt.append("<|im_end|>")
        self.llmConfig.antiPrompt.append("<|im_start|>")
        self.llmConfig.n_predict = 128
        self.llmConfig.mirostat = 2
        self.llmConfig.frequency_penalty = 0.8
        self.llmConfig.top_p = 0.60
        self.llmConfig.top_k = 25
        self.llmConfig.temperature = 0.75
        self.llmConfig.nOffloadLayer = 100
        self.llmConfig.mainGPU = 0
        self.llmConfig.repeat_penalty = 1.2

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

    def prompt(self, person: str, message: str, sysPromptSetting: str) -> str:
        message = self.preprocessMessage(message=message)
        # TODO: Change system prompts from personality to actual explanations of behaviour
        systemPrompt = self.accessMemories(keyword=sysPromptSetting, setting='systemPrompt')
        # TODO: Use current systemPrompt style as the personality description
        personality = self.accessMemories(keyword="personality", setting="melba")
        # TODO: Implement retrieval augmented generation aka get relevant information into the context
        context = self.situationalContext()
        # old

        characterInformation = self.accessMemories(keyword=person, setting="characterdata")
        generalInformation = f"{self.getGeneralInformation(message=message)}\n" # TODO: Make it happen
        pastConversation = self.accessMemories(keyword=person, setting='savedchat')

        self.convoStyle = self.llm.promptTemplate(inputText=message)

        finalPrompt = (f"{self.llm.systemPromptPrefix}\n{systemPrompt}" +
                       f"{characterInformation}\n" +
                       f"{generalInformation}" +                               # TODO: implement information retrieval
                       f"{self.llm.systemPromptSplitter}\n{pastConversation}\n" + # TODO: for certain keywords
                          self.convoStyle)

        finalPrompt = finalPrompt.replace("{llmName}", self.llmConfig.modelName)
        return finalPrompt

    def getMelbaResponse(self, message, sysPromptSetting, person, stream=False) -> str:
        self.curPrompt = self.prompt(person,
                                              message,
                                              sysPromptSetting)        # insert model specific tokens
        self.llm.loadPrompt(path=None, prompt=self.curPrompt, type=self.llmConfig.modelType)
        if self.curPrompt == "":  # we shouldn't even get here
            print("melbaToast: Something went wrong while constructing the prompt, please restart Melba.")

        print(f"\nmelbaToast: Current prompt is:\n -[{self.curPrompt}]-\n")

        response = self.llm.response(stream=False)

        filteredResponse = self.filterMessage(response)

        self.updateMemory(type="savedchat", person=person,
                          newContent=(f"{self.accessMemories(keyword=person, setting='savedchat')}\n"
                                      + self.convoStyle + response + self.llm.inputSuffix))
        self.llm.reset()  # needs to be improved
        self.log(message=f"User: '{person}' \tMessage: '{message}' \tMelba response: '{response}' \tEmotion: "
                         f"'{self.curEmotion}'")

        actualResponse = {'response' : filteredResponse, 'emotions' : self.emotion(response)}
        return json.dumps(actualResponse)

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

    def getSavedChat(self, username: str):
        response = self.memoryDB.metadataQueryDB(type="savedchat", identifier=username)

        if response == "":
            self.logger.log(message=f"melbaToast: No saved chat with username {username} found.")
        else:
            newlines = response.count('\n')
            if newlines >= 8:
                new = '\n'.join(response.split('\n')[5:])
                self.updateMemory(type="savedchat", identifier=username, newContent=new)
                response = new

        return response

    def getSystemprompt(self, queries: List[str]):
        response = json.loads(
            json.dumps(self.memoryDB.vectorQueryDB(queries=queries, filter={"type": {"$eq": "systemPrompt"}}))
        )
        response = response['documents']

        if response == "":
            self.logger.log(message=f"melbaToast: No system prompt with queries '{queries}' found, loading generic.")
            response = self.memoryDB.metadataQueryDB(type="systemPrompt", identifier="generic")

        return response[0][0]

    def getPersonalInformation(self, name: str):
        response = self.memoryDB.metadataQueryDB(type="characterdata", identifier=name)

        if response == "":
            self.logger.log(message=f"melbaToast: No personal information about {name} found.")

        return response

    def updateMemory(self, type: str, identifier: str, newContent: str):
        self.memoryDB.updateOrCreateDBEntry(type=type, identifier=identifier, content=newContent)
        self.logger.log(message=f"melbaToast: Updated memory entry of type '{type}' "
                                f"with identifier '{identifier}' and content '{newContent}'")

    def wipeDB(self):
        self.memoryDB.chromaClient.reset()
        self.logger.log(message="melbaToast: MemoryDB fully wiped.")

    # TODO: delete this function later on
    def getGeneralInformation(self, message: str) -> str:
        topN = 3
        keywordExtractor = yake.KeywordExtractor()
        keywords = keywordExtractor.extract_keywords(text=message)
        keywords.sort(key=lambda e: e[1])
        keywords = keywords[-topN:]
        topKeywords = []
        for kwPair in keywords:
            topKeywords.append(kwPair[0])
        generalInfo = json.loads(json.dumps(self.memoryDB.vectorQueryDB(queries=topKeywords)))
        generalInfo = generalInfo['documents'][0][0]
        return generalInfo

    def accessMemories(self, keyword: str, setting: str) -> str:
        if setting == "savedchat":
            res = self.getSavedChat(username=keyword)
        elif setting == "systemPrompt":
            res = self.getSystemprompt(queries=[keyword])
        elif setting == "characterdata":
            res = self.getPersonalInformation(name=keyword)
        elif setting == "generalinformation":  # TODO: delete this setting later on
            res = self.getGeneralInformation(keyword=keyword)
        else:
            self.logger.log(message=f"melbaToast: Setting {setting} not found.")
            res = ""

        return res


class Context:
    def __init__(self, memoryDB: MemoryDB):
        self.memoryDB = memoryDB

    # TODO: increase search accuracy and response volume (reimplementation)
    def returnWebContent(self, searchQuery: str = None) -> List[str]:
        return ["placeholder"]

    def situationalContext(self, message: str) -> str:
        # stage 2(?) should enable a llm summarizing and formatting the message into a useful query for the vectordb
        vectorstorageresponse = self.memoryDB.vectorQueryDB(queries=[message], filter=None, nResults=2)
        # stage 2 should summarize the results into something usable           - needs to be filtered
        if vectorstorageresponse == "":
            webResponse = self.returnWebContent(searchQuery=message) # searchQuery will be extracted keywords
        return "placeholder"

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
        wordsum = 0                                                                  # accurate and performant

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


class Utils:
    def __init__(self, memoryDB: MemoryDB):
        self.memoryDB = memoryDB
        self.swearWords = ""  # TODO: list instead of string
        self.maliciousWords = ""  # TODO: List instead of string

    def isSwearWord(self, word: str) -> bool:
        if self.swearWords == "":
            self.swearWords = (self.memoryDB.metadataQueryDB(type="swearwords", identifier="all")).split()
        if word in self.swearWords:
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
            except:
                FileExistsError("File does not exist.")

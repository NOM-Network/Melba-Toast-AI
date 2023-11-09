import LLMCore
import memoryDB
from dataclasses import dataclass
from datetime import datetime
from nrclex import NRCLex
from typing import List
import requests
import math
import json
import yake

@dataclass
class strIntPair:
    string: str
    integer: int

# TODO: Handle system prompts inside the databank
class Melba:
    def __init__(self, modelPath: str, databasePath: str, logpath: str = None, backupPath: str = None):
        self.llmConfig = self.defaultConfig()

        self.llmConfig.modelPath = modelPath
        self.backupPath = backupPath
        self.logPath = logpath

        self.stage = 0

        self.curEmotion = "neutral"
        self.swearWords = ""
        self.maliciousWords = ""

        self.memoryDB = memoryDB.MemoryDB(databasePath)
        self.llm = LLMCore.LlamaModel(self.llmConfig)
        self.backup = False
        self.llm.loadPrompt(type=self.llmConfig.modelType)

        self.log(message="Initialized Melba.")

    def setBackup(self, mode: bool):
        if mode == True and self.backup == False:
            self.llm.exit()
            self.llm = LLMCore.LlamaOrig(self.llmConfig)
            self.llm.loadPrompt(type=self.llmConfig.modelType)
        elif mode == False and self.backup == True:
            self.llm = LLMCore.LlamaModel(self.llmConfig)

    def defaultConfig(self):
        llmConfig = LLMCore.defaultLlamactxParams()
        llmConfig.nCtx = 1024
        llmConfig.n_keep = 1024
        llmConfig.modelName = "Melba"
        llmConfig.modelType = "openhermes-mistral"
        #llmConfig.antiPrompt.append("You:")
        llmConfig.antiPrompt.append("<|im_end|>")
        llmConfig.antiPrompt.append("<|im_start|>")
        llmConfig.antiPrompt.append("<br>")
        llmConfig.antiPrompt.append("!!")
        llmConfig.antiPrompt.append("<")
        #llmConfig.antiPrompt.append(".")
        #llmConfig.antiPrompt.append("?")
        llmConfig.n_predict = 128
        llmConfig.mirostat = 2
        llmConfig.frequency_penalty = 0.8
        llmConfig.top_p = 0.60
        llmConfig.top_k = 25
        llmConfig.temperature = 0.75
        llmConfig.nOffloadLayer = 100
        llmConfig.mainGPU = 0
        llmConfig.repeat_penalty = 1.2

        return llmConfig

    def updateLLMConfig(self, newConfig):
        cfg = json.loads(newConfig)
        newParm = self.getCurrentConfig()
        attributes = ["n_keep", "n_predict", "tfs_z", "typical_p",
                      "top_k", "top_p", "temperature", "mirostat",
                      "mirostat_tau", "mirostat_eta", "repeat_last_n", "repeat_penalty",
                      "frequency_penalty", "presence_penalty", "penalize_nl", "n_batch"]

        for attribute in attributes:
            setattr(newParm, attribute, cfg[attribute])

        self.llm.update(newParm)

    def getCurrentConfig(self):
        return self.llmConfig

    # TODO: let this function also modify general memories
    def updateMemory(self, type: str, person: str, newContent: str):
        self.memoryDB.updateOrCreateDBEntry(type=type, identifier=person, content=newContent)
        self.log(message=f"Updated memory entry of type '{type}' with identifier '{person}' and content '{newContent}'")

    # each memory access should have its own function for future updates
    # this code isn't very dry ):
    def getSavedChat(self, username: str):
        response = self.memoryDB.metadataQueryDB(type="savedchat", identifier=username)

        if response == "":
            print(f"melbaToast: No saved chat with username {username} found.")
        else:
            newLines = response.count('\n')
            if newLines >= 8:
                new = '\n'.join(response.split('\n')[5:])
                self.updateMemory(type="savedchat", person=username, newContent=new)
                response = new
        return response

    def getSystemprompt(self, queries: List[str]):
        response = json.loads(
                    json.dumps(self.memoryDB.vectorQueryDB(queries=queries, filter={"type" : {"$eq" : "systemPrompt"}}))
                    )
        response = response['documents']
        if response == "":
            print(f"melbaToast: No system prompt with queries '{queries}' found, loading generic.")
            response = self.memoryDB.metadataQueryDB(type="systemPrompt", identifier="generic")
        return response[0][0]

    def getPersonalInformation(self, name: str):
        response = self.memoryDB.metadataQueryDB(type="characterdata", identifier=name)

        if response == "":
            print(f"melbaToast: No personal information about {name} found.")
        return response

    # results are probably not very accurate and need to be improved
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
        print(f"melbaToast debug: Inside getPastMemories keyword: {keyword} setting: {setting}")
        res = -1
        if setting == "savedchat":
            res = self.getSavedChat(username=keyword)
        elif setting == "systemPrompt":
            res = self.getSystemprompt(queries=[keyword])
        elif setting == "characterdata":
            res = self.getPersonalInformation(name=keyword)
        elif setting == "generalinformation":
            res = self.getGeneralInformation(keyword=keyword)
        else:
            print(f"melbaTost: Setting {setting} not found.")
            res = ""

        return res

    # this function isn't finished
    # implement webscraping at some point too
    # for now the raw results & the google answer cards(need to be read too) should suffice
    def returnWebContent(self, searchQuery: str = None) -> List[str]:
        if searchQuery is None:
            return []
        results = []

        searchParams = {'q' : searchQuery, 'format' : 'json'}
        response = json.loads((requests.get("http://localhost:2000/search")).text)

        if response['results'] is None:
            return []
        for r in response['results']:
            results.append(r['content'])

        return results

    def situationalContext(self, message: str) -> str:
        # stage 2(?) should enable a llm summarizing and formatting the message into a useful query for the vectordb
        vectorStorageResponse = self.memoryDB.vectorQueryDB(queries=[message], filter=None, nResults=2)
        # stage 2 should summarize the results into something usable           - needs to be filtered
        if vectorStorageResponse == "":
            webResponse = self.returnWebContent(searchQuery=message) # searchQuery will be extracted keywords
        return "placeholder"

    def isSwearWord(self, word: str) -> bool:
        if self.swearWords == "":
            self.swearWords = (self.memoryDB.metadataQueryDB(type="swearwords", identifier="all")).split()
        if word in self.swearWords:
            return True
        return False

    def maliciousWordsCount(self, words: List[str]) -> int:
        if self.maliciousWords == "":
            self.maliciousWords = self.memoryDB.metadataQueryDB(type="maliciouswords", identifier="all").split()
        mWordCount = 0
        for word in words:
            if word in self.maliciousWords:
                mWordCount += 1
        return mWordCount

    def characterFrequency(self, sentence: str) -> List[strIntPair]:
        charFreq: List[strIntPair] = []

        for c in sentence:
            exists = False
            for pair in charFreq:
                if pair.string == c:
                    pair.integer += 1
                    exists = True
            if not exists:
                pair = strIntPair(c, 1)
                charFreq.append(pair)
        return charFreq

    def characterProbability(self, frequencies: List[strIntPair], target: str):  # this function should likely be
        targetIndex = 0                                                          # swapped out for something more
        sum = 0                                                                  # accurate and performant

        iterator = 0
        for field in frequencies:
            sum += field.integer
            targetIndex = iterator if field.string == target else targetIndex
            iterator += 1

        return frequencies[targetIndex].integer/sum

    def sentenceEntropy(self, sentence: str):
        frequencies: List[strIntPair] = self.characterFrequency(sentence=sentence)
        entropy = 0.0

        for field in frequencies:
            charProb = self.characterProbability(frequencies=frequencies, target=field.string)
            entropy += charProb * math.log2(charProb)

        return -entropy

    def preprocessMessage(self, message: str) -> str:
        if self.sentenceEntropy(sentence=message) > 3.0 and\
           self.maliciousWordsCount(words=message.split()) <= 0:
            return message
        return ""
        # stage 1 will include a llm preprocessing this message and further deciding whether it is valid for
        # further use
        # stage 0 will use this function, though a better filter will be needed

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

    def emotion(self, text):
        n = NRCLex(text=text)
        currentEmotions = []

        for emotion in n.top_emotions:
            currentEmotions.append(emotion[0])

        return currentEmotions

    def getEmotion(self):
        return self.curEmotion

    def filterMessage(self, message: str) -> str:
        #TODO: Filter excessive emotes
        #TODO: Shorten sentences and add more punctiation.
        filteredMessage = []
        for word in message.split():
            print(word)
            if self.isSwearWord(word=word):
                filteredMessage.append("[TOASTED]")
            else:
                filteredMessage.append(word)

        return ' '.join(filteredMessage)

    def getMelbaResponse(self, message, sysPromptSetting, person, stream=False) -> str:
        self.curPrompt = self.prompt(person,
                                              message,
                                              sysPromptSetting)        # insert model specific tokens
        self.llm.loadPrompt(path=None, prompt=self.curPrompt, type=self.llmConfig.modelType)
        if self.curPrompt == "":  # we shouldn't even get here
            print("melbaToast: Something went wrong while constructing the prompt, please restart Melba.")

        # if stream: # streaming disabled for now
        #     for token in self.llm.response(prompt=self.curPrompt, stream=True):
        #         yield token
        # else:
        print(f"\nmelbaToast: Current prompt is:\n -[{self.curPrompt}]-\n")

        response = self.llm.response(stream=False)

        # filter
        filteredResponse = self.filterMessage(response)

        self.updateMemory(type="savedchat", person=person,
                          newContent=(f"{self.accessMemories(keyword=person, setting='savedchat')}\n"
                                      + self.convoStyle + response + self.llm.inputSuffix))
        self.llm.reset()  # needs to be improved
        self.log(message=f"User: '{person}' \tMessage: '{message}' \tMelba response: '{response}' \tEmotion: "
                         f"'{self.curEmotion}'")

        actualResponse = {'response' : filteredResponse, 'emotions' : self.emotion(response)}
        return json.dumps(actualResponse)
    # stream: Whether to return full response or stream the result
    def regenerateResponse(self, stream=False):
        if stream:
            for token in self.llm.response():
                yield token
        else:
            return self.llm.response()

    def wipeDB(self):
        print("melbaToast: This action will completely wipe the MemoryDB, are you sure you want to continue?y/n")
        choice = input()
        if choice == 'y':
            self.memoryDB.chromaClient.reset()

    def log(self, message: str):
        if self.logPath is not None:
            try:
                with open(self.logPath, mode="a") as file:
                    file.write(f"\n[{datetime.utcnow().strftime('%y-%m-d %H:%M:%S')}]-[{message}]")
            except:
                FileExistsError

    def end(self):
        if self.backupPath is not None:
            self.memoryDB.backupDB(backupPath=self.backupPath)
        self.llm.exit()

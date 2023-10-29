import LLMCore
import memoryDB
from nrclex import NRCLex
from datetime import datetime
import time
import json

# TODO: Handle system prompts inside the databank
class Melba:
    def __init__(self, modelPath: str, databasePath: str, logpath: str = None, backupPath: str = None):
        self.llmConfig = self.defaultConfig()

        self.llmConfig.modelPath = modelPath
        self.backupPath = backupPath
        self.logPath = logpath

        self.memoryDB = memoryDB.MemoryDB(databasePath)
        self.llm = LLMCore.LlamaModel(self.llmConfig)
        self.llm.loadPrompt(type=self.llmConfig.modelType)
        self.curEmotion = "neutral"
        self.curPrompt = ""
        self.swearWords = []
        self.log(message="Initalized Melba.")

    def defaultConfig(self):
        llmConfig = LLMCore.defaultLlamactxParams()
        llmConfig.nCtx = 1024
        llmConfig.n_keep = 1024
        llmConfig.modelName = "Melba"
        llmConfig.modelType = "OpenHermes-Mistral"
        llmConfig.antiPrompt.append("You:")
        llmConfig.antiPrompt.append("Melba:")
        llmConfig.n_predict = 32
        llmConfig.mirostat = 2
        llmConfig.frequency_penalty = 8
        llmConfig.top_p = 0.60
        llmConfig.top_k = 25
        llmConfig.temperature = 0.50
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
            if newLines >= 12:
                new = '\n'.join(response.split('\n')[4:])
                self.updateMemory(type="savedchat", person=username, newContent=new)
                response = new
        return response

    def getSystemprompt(self, keyword: str):
        response = self.memoryDB.metadataQueryDB(type="systemprompt", identifier=keyword)

        if response == "":
            print(f"melbaToast: No system prompt with keyword {keyword} found, loading generic.")
            response = self.memoryDB.metadataQueryDB(type="systemprompt", identifier="generic")
        return response

    def getPersonalInformation(self, name: str):
        response = self.memoryDB.metadataQueryDB(type="characterdata", identifier=name)

        if response == "":
            print(f"melbaToast: No personal information about {name} found.")
        return response

    # results are probably not very accurate and need to be improved
    def getGeneralInformation(self, keyword: str):
        response = self.memoryDB.vectorQueryDB(keyword)

        if response == "":
            print(f"melbaToast: No information about {keyword} was found.")
        return response

    def accessMemories(self, keyword: str, setting: str) -> str:
        print(f"melbaToast debug: Inside getPastMemories keyword: {keyword} setting: {setting}")
        res = -1
        if setting == "savedchat":
            res = self.getSavedChat(username=keyword)
        elif setting == "systemPrompt":
            res = self.getSystemprompt(keyword=keyword)
        elif setting == "characterdata":
            res = self.getPersonalInformation(name=keyword)
        elif setting == "generalinformation":
            res = self.getGeneralInformation(keyword=keyword)
        else:
            print(f"melbaTost: Setting {setting} not found.")
            res = ""

        return res

    def isSwearWord(self, word: str) -> bool:
        if self.swearWords == "":
            self.swearWords = (self.memoryDB.metadataQueryDB(type="swearwords", identifier="all")).split()
            print("in swearwordinit")
        if word in self.swearWords:
            return True
        return False

    def structurePrompt(self, person: str, message: str, sysPromptSetting: str) -> str:
        systemPrompt = self.accessMemories(keyword=sysPromptSetting, setting='systemPrompt')
        characterInformation = self.accessMemories(keyword=person, setting="characterdata")
        generalInformation = f"The current data is {datetime.today().strftime('%Y-%m-%d')}\n" \
                             f"The current time is {time.strftime('%H:%M:%S', time.localtime())}" # TODO: Make it happen
        pastConversation = self.accessMemories(keyword=person, setting='savedchat')

        self.convoStyle = self.llm.promptTemplate()
        self.convoStyle = self.convoStyle.replace("[inputName]", person).replace("[outputName]", self.llmConfig.modelName)
        self.convoStyle = self.convoStyle.replace("[inputText]", message)

        finalPrompt = (f"{self.llm.systemPromptPrefix}{systemPrompt}" +
                       f"{characterInformation}" +
                       f"{characterInformation}\n" +
                       f"{generalInformation}\n" +                               # TODO: implement information retrieval
                       f"{self.llm.systemPromptSplitter}\n{pastConversation}\n" + # TODO: for certain keywords
                          self.convoStyle)

        finalPrompt = finalPrompt.replace("{llmName}", self.llmConfig.modelName)
        return finalPrompt

    def emotion(self, text):
        n = NRCLex(text=text)
        rawEmotions = n.top_emotions
        currentEmotions = []

        print(rawEmotions)
        if rawEmotions[0][1] == 0.0:
            return "neutral"

        for emotion in rawEmotions:
            currentEmotions.append(emotion[0])

        if 'surprise' and 'positive' in currentEmotions:
            return 'happy'
        elif 'surprise' and 'negative' in currentEmotions:
            return 'angered'
        elif 'fear' and 'negative' in currentEmotions:
            return 'scared'
        elif 'positive' or 'joy' in currentEmotions:
            return "happy"
        elif 'anger' or 'negative' in currentEmotions:
            return 'angered'
        elif 'sadness' or 'negative' in currentEmotions:
            return 'sad'
        elif 'trust' in currentEmotions:
            return 'trust'
        elif 'fear' in currentEmotions:
            return 'scared'
        elif 'anticipation' in currentEmotions:
            return 'neutral'
        return 'neutral'

    def getEmotion(self):
        return self.curEmotion

    def getMelbaResponse(self, message, sysPromptSetting, person, stream=False) -> str:
        self.curPrompt = self.structurePrompt(person,
                                              message,
                                              sysPromptSetting)        # insert model specific tokens
        self.llm.loadPrompt(path=None, prompt=self.curPrompt, type="openhermes-mistral")
        if self.curPrompt == "":  # we shouldn't even get here
            print("melbaToast: Something went wrong while constructing the prompt, please restart Melba.")

        # if stream: # streaming disabled for now
        #     for token in self.llm.response(prompt=self.curPrompt, stream=True):
        #         yield token
        # else:
        print(f"\nmelbaToast: Current prompt is:\n -[{self.curPrompt}]-\n")

        response = (self.llm.response(stream=False)).split()
        #response = self.llm.response(stream=False)

        # filter
        actualResponse = ""
        for word in response:
            if self.isSwearWord(word=word):
                actualResponse += " [TOASTED]"
            else:
                actualResponse += ' ' + word

        self.updateMemory(type="savedchat", person=person,
                          newContent=(f"{self.accessMemories(keyword=person, setting='savedchat')}\n"
                                      + self.convoStyle + actualResponse + self.llm.inputPostfix))
        self.llm.reset()  # needs to be improved
        self.curEmotion = self.emotion(actualResponse)
        self.log(message=f"User: '{person}' \tMessage: '{message}' \tMelba response: '{actualResponse}' \tEmotion: "
                         f"'{self.curEmotion}'")
        return actualResponse
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

import LLMCore
import memoryDB
from nrclex import NRCLex
import json

# TODO: Handle system prompts inside the databank
class Melba:
    def __init__(self, modelPath, systemPromptPath, databasePath, backupPath = None):
        self.llmConfig = self.defaultConfig()

        self.llmConfig.modelPath = modelPath
        self.systemPromptPath = systemPromptPath
        self.backupPath = backupPath

        self.memoryDB = memoryDB.MemoryDB(databasePath)
        self.llm = LLMCore.LlamaModel(self.llmConfig)
        self.llm.loadPrompt(type=self.llmConfig.modelType)
        self.curEmotion = "neutral"
        self.curPrompt = ""

    def defaultConfig(self):
        llmConfig = LLMCore.defaultLlamactxParams()
        llmConfig.modelName = "Melba"
        llmConfig.modelType = "OpenHermes-Mistral"
        llmConfig.antiPrompt.append("You:")
        llmConfig.antiPrompt.append("Melba:")
        llmConfig.n_predict = 32
        llmConfig.mirostat = 2
        llmConfig.frequency_penalty = 8
        llmConfig.top_p = 0.45
        llmConfig.top_k = 25
        llmConfig.temperature = 0.80
        llmConfig.nOffloadLayer = 0
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

    # each memory access should have its own function for future updates
    # this code isn't very dry ):
    def getSavedChat(self, username: str):
        response = self.memoryDB.metadataQueryDB(type="savedchat", identifier=username)

        if response == "":
            print(f"melbaToast: No saved chat with username {username} found.")
        else:
            newLines = response.count('\n')
            if newLines >= 12:
                new = '\n'.join(response.split('\n')[:4])
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

    def structurePrompt(self, person: str, message: str, sysPromptSetting: str) -> str:
        systemPrompt = self.accessMemories(keyword=sysPromptSetting, setting='systemPrompt')
        characterInformation = self.accessMemories(keyword=person, setting="characterdata")
        characterInformation = characterInformation if characterInformation != "" else '\n'
        #generalInformation = self.getPastMemories(keyword='twitchchatter', setting='generalinformation') # TODO: Make it happen
        pastConversation = self.accessMemories(keyword=person, setting='savedchat')

        self.convoStyle = self.llm.promptTemplate()
        self.convoStyle = self.convoStyle.replace("[inputName]", person).replace("[outputName]", self.llmConfig.modelName)
        self.convoStyle = self.convoStyle.replace("[inputText]", message)

        finalPrompt = (f"{self.llm.systemPromptPrefix}{systemPrompt}" +
                       f"{characterInformation}" +
                       f"{characterInformation}\n" +
                       #f"{generalInformation}\n" +                               # TODO: implement information retrieval
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
            return 'angerd'
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
        self.llm.loadPrompt(path=None, prompt=self.curPrompt, type="pygmalion")
        if self.curPrompt == "":  # we shouldn't even get here
            print("melbaToast: Something went wrong while constructing the prompt, please restart Melba.")

        # if stream: # streaming disabled for now
        #     for token in self.llm.response(prompt=self.curPrompt, stream=True):
        #         yield token
        # else:
        print(f"\nmelbaToast: Current prompt is:\n -[{self.curPrompt}]-\n")

        response = self.llm.response(stream=False)

        self.updateMemory(type="savedchat", person=person,
                          newContent=(f"{self.accessMemories(keyword=person, setting='savedchat')}\n"
                                      + self.convoStyle + response + self.llm.inputPostfix)) # there is definitely a
        self.curEmotion = self.emotion(response)                                            # more elegant solution

        return response
    # stream: Whether to return full response or stream the result
    def regenerateResponse(self, stream=False):
        if stream:
            for token in self.llm.response():
                yield token
        else:
            return self.llm.response()

    def end(self):
        if self.backupPath is not None:
            self.memoryDB.backupDB(backupPath=self.backupPath)
        self.llm.exit()

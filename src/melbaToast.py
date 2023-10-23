import LLMCore
import memoryDB
from nrclex import NRCLex
import json

# TODO: Take in JSON objects as function arguments
# TODO: Handle system prompts inside the databank too
class Melba:
    def __init__(self, modelPath, systemPromptPath, databasePath, backupPath = None):
        self.llmConfig = self.defaultConfig()
        self.llmConfig.modelPath = modelPath
        self.systemPromptPath = systemPromptPath
        self.backupPath = backupPath

        self.curPrompt = None
        self.experimental = False

        self.curEmotion = "neutral"
        self.memoryDB = memoryDB.MemoryDB(databasePath)
        self.llm = LLMCore.LlamaModel(self.llmConfig)

    def defaultConfig(self):
        llmConfig = LLMCore.defaultLlamactxParams()
        llmConfig.modelName = "Melba"
        llmConfig.modelType = "OpenHermes-Mistral"
        llmConfig.antiPrompt.append("You:")
        llmConfig.antiPrompt.append("Melba:")
        llmConfig.n_predict = 64
        llmConfig.mirostat = 2
        llmConfig.frequency_penalty = 8
        llmConfig.top_p = 0.50
        llmConfig.top_k = 20
        llmConfig.temperature = 0.80
        llmConfig.nOffloadLayer = -1 # GPU offloading is broken at the moment, will fix ASAP
        llmConfig.mainGPU = 1
        llmConfig.repeat_penalty = 1.2

        return llmConfig

    def getCurrentConfig(self):
        return self.llmConfig

    def updateLLMConfig(self, newConfig):
        cfg = json.loads(newConfig)
        newParm = self.getCurrentConfig()

        newParm.n_keep = newConfig["n_keep"]
        newParm.n_predict = newConfig["n_predict"]
        newParm.tfs_z = newConfig["tfs_z"]
        newParm.typical_p = newConfig["typical_p"]
        newParm.top_k = newConfig["top_k"]
        newParm.top_p = newConfig["top_p"]
        newParm.temperature = newConfig["temperature"]
        newParm.mirostat = newConfig["mirostat"]
        newParm.mirostat_tau = newConfig["mirostat_tau"]
        newParm.mirostat_eta = newConfig["mirostat_eta"]
        newParm.repeat_last_n = newConfig["repeat_last_n"]
        newParm.repeat_penalty = newConfig["repeat_penalty"]
        newParm.frequency_penalty = newConfig["frequency_penalty"]
        newParm.presence_penalty = newConfig["presence_penalty"]
        newParm.penalize_nl = newConfig["penalize_nl"]
        newParm.n_batch = newConfig["n_batch"]

        self.llm.update(newParm)

    # TODO: let this function also modify general memories
    def updateMemory(self, person: str, newContent: str):
        self.memoryDB.updateOrCreateDBEntry(type="savedchat", identifier=person, content=newContent)

    # settings depend on the system prompts file
    # 0 - Generic
    # 1 - For Twitch chatter
    # 2 - Individual (For collabs)
    def getSystemPrompt(self, filepath: str, setting) -> str:
        prompt = ""
        counter = 0

        try:
            with open(filepath) as file:
                for line in file:
                    if counter == setting:
                        prompt += line
                    if line.find('-=-') != -1: # -=- is used to seperate different system prompts
                        counter += 1
        except FileExistsError:
            print("Invalid filepath")

        if prompt == "":
            print(f"melbaToast: No system prompt with the setting {setting} found.")
            return ""

        print("melbaToast debug: Inside getSystemPrompt")
        return prompt

    def getPastMemories(self, keyword: str, setting: str) -> str:
        print(f"melbaToast debug: Inside getPastMemories keyword: {keyword} setting: {setting}")
        res = -1
        if setting == "savedchat":
            res = self.memoryDB.metadataQueryDB(type="savedchat", identifier=keyword)
        elif setting == "personinformation":
            res = self.memoryDB.metadataQueryDB(type="personinformation", identifier=keyword)
        elif setting == "generalinformation":
            res = self.memoryDB.vectorQueryDB(keyword)
        else:
            print(f"melbaTost: Setting {setting} not found.")
            res = ""

        if res == -1:
            print("melbaToast: Memory not found.")
            res = ""


        return res

    def structurePrompt(self, person: str, message: str, sysPromptSetting: int, sysPromptToken: str, sysPromptSplitter: str) -> str:
        information = ""            # experimental information retrieval

        if self.experimental:
            words = []
            temp = ""

            for character in message:
                if character == ' ' and temp != "":
                    words.append(temp)
                    temp = ""
                else:
                    temp += character

            for word in words:
                res = self.memoryDB.metadataQueryDB(type="infoaboutperson", identifier=word) # will be replaced with a vector db query
                if res:
                    information = res # experimental information retrieval

        pastConversation = self.getPastMemories(keyword=person, setting='savedchat')
        newLines = pastConversation.count('\n')
        if newLines >= 8:
            new = '\n'.join(pastConversation.split('\n')[:4])
            self.updateMemory(person=person, newContent=new)
            pastConversation = new

        systemPrompt = self.getSystemPrompt(self.systemPromptPath, sysPromptSetting)
        #generalInformation = self.getPastMemories(keyword='twitchchatter', setting='generalinformation') # TODO: Make it happen
        if self.llm.parameters.modelType == "OpenHermes-Mistral":
            print("debug")
            self.convoStyle = '\n' + self.llm.inputPrefix + "user\n{person}: " + message + self.llm.outputPrefix + '\n' + \
                              self.llm.inputPrefix + "assistant\n"
        else:
            self.convoStyle = "{person}: " + message + "\n{llmName}: "

        finalPrompt = (f"{sysPromptToken}{systemPrompt}" +
                       f"{self.getPastMemories(keyword=person, setting='personinformation')}" +
                       ((f"{information}\n") if self.experimental else "") +
                       #f"{generalInformation}\n" +                 # TODO: implement information retrieval
                       f"{sysPromptSplitter}\n{pastConversation}" + # TODO: for certain keywords
                          self.convoStyle)                               # TODO: Insert prefix for both user input and llm response


        finalPrompt = finalPrompt.replace("{llmName}", "Melba").replace("{person}", person).replace("-=-", "") # this shouldn't even be needed in the first place

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

    # stream: If True, stream the response(Generator object)
    # sysPromptSetting: 0 = generic, 1 = single person(viewer), 2 = individual person
    # person: The name or username of the person which Melba is responding to
    # message: Text which Melba will respond to
    def getMelbaResponse(self, message, sysPromptSetting, person, stream=False) -> str:
        self.curPrompt = self.structurePrompt(person,
                                              message,
                                              sysPromptSetting,
                                              self.llm.systemPromptPrefix,
                                              self.llm.systemPromptSplitter)        # insert model specific tokens
        self.llm.loadPrompt(path=None, prompt=self.curPrompt, type="pygmalion")

        # if stream: # streaming disabled for now
        #     for token in self.llm.response(prompt=self.curPrompt, stream=True):
        #         yield token
        # else:
        print(f"\nmelbaToast: Current prompt is:\n{self.curPrompt}\n")

        response =  self.llm.tempGenerate()

        self.updateMemory(person, (f"{self.getPastMemories(keyword=person, setting='savedchat')}\n" + self.convoStyle + response))
        #self.updateMemory(person, (f"{self.getPastMemories(keyword=person, setting='savedchat')}\n" + "{person}: " + message + "\n{llmName}: " + response + '\n'))
        self.curEmotion = self.emotion(response)

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

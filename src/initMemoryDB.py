import memoryDB
from typing import List

memDB = memoryDB.MemoryDB(path="db")

def initSysPrompts(filePath: str):
    systemPrompts: List[str] = []
    curPrompt = ""

    with open(filePath) as sysPromptFile:
        for line in sysPromptFile.readlines():
            if line.find("-=sysPromptSplitter=-") != -1:
                systemPrompts.append(curPrompt)
                curPrompt = ""
            else:
                curPrompt += line
        if curPrompt != '':
            systemPrompts.append(curPrompt)

    for prompt in systemPrompts:
        memDB.newDBEntry(type="systemPrompt", identifier="generic", content=prompt)

def initPersonalityPrompts(filePath: str):
    personalities: List[str] = []
    personalityIdentifiers: List[str] = []
    curPersonality = ""
    curIdentifier = ""

    with open(filePath) as personalitiesFile:
        for line in personalitiesFile.readlines():
            #print(line)
            if line.find("-=personalitySplitter=-") != -1:
                print(curPersonality)
                print(curIdentifier)
                personalities.append(curPersonality.replace('\n', ''))
                personalityIdentifiers.append(curIdentifier.replace('\n', ''))
                curPersonality = ""
                curIdentifier = ""
            elif line.find("-=personalityStart=-") != -1:
                curIdentifier = line[20:]
                #print(curIdentifier)
            else:
                curPersonality += line
        if curPersonality != '':
            personalities.append(curPersonality.replace('\n', ''))
            personalityIdentifiers.append(curIdentifier.replace('\n', ''))
        for personality in personalities:
            memDB.newDBEntry(type="personality",
                             identifier=personalityIdentifiers[personalities.index(personality)],
                             content=personality)

def initInformationMemory(filePath: str):
    information: List[str] = []
    informationHeaders: List[str] = []
    curInformation = ""
    curHeader = ""

    with open(filePath) as informationFile:
        for line in informationFile.readlines():
            if line.find("-=informationSplitter=-") != -1:
                information.append(curInformation.replace("\n", ""))
                informationHeaders.append(curHeader.replace("\n", ""))
                curInformation = ""
                curHeader = ""
            elif line.find("-=informationStart=-") != -1:
                curHeader = line[20:]
            else:
                curInformation += line
        if curInformation != '':
            information.append(curInformation.replace("\n", ""))
            informationHeaders.append(curHeader.replace("\n", ""))
    for header in informationHeaders:
        memDB.newDBEntry(type="information", identifier=header,
                         content=information[informationHeaders.index(header)])

def initSwearWords(filePath: str, filePathExclusions: str = None):
    with open(filePath) as swearWords:
        swearWordsFull = swearWords.read()
    if filePathExclusions is not None:
        with open(filePathExclusions):
            for line in swearWords.readlines():
                swearWordsFull = swearWordsFull.replace(line, "")
    if swearWordsFull != "":
        memDB.newDBEntry(type="swearwords", identifier="all", content=swearWordsFull)

initSysPrompts(filePath="memories/systemPrompts.txt")
initPersonalityPrompts(filePath="memories/personalities.txt")
initCharacterMemory(filePath="memories/characterInformation.txt")
initSwearWords(filePath="memories/bannedWords.txt")
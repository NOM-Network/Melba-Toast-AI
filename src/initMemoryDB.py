import memoryDB
from typing import List

memDB = memoryDB.MemoryDB(path="insert path to db")

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
            if line.find("-=personalitySplitter=-") != -1:
                print(curPersonality)
                print(curIdentifier)
                personalities.append(curPersonality.replace('\n', ''))
                personalityIdentifiers.append(curIdentifier.replace('\n', ''))
                curPersonality = ""
                curIdentifier = ""
            elif line.find("-=personalityStart=-") != -1:
                curIdentifier = line[20:]
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
    content = []
    header = ""
    tempContent = []

    with open(filePath) as file:
        for line in file.readlines():
            if line.find("-=informationSplitter=-") != -1:
                for c in tempContent:
                    content.append([header, c])
                tempContent = []
                header = ""
            elif line.find("-=informationStart=-") != -1:
                header = line[20:].replace('\n', '')
            else:
                tempContent.append(line.replace('\n', ''))

    i = 0
    for information in content:
        memDB.newDBEntry(type="information", identifier=f"{information[0]}{i}",
                         content=information[1])
        i += 1

def initSwearWords(filePath: str, filePathExclusions: str = None):
    with open(filePath) as swearWords:
        swearWordsFull = swearWords.read()
    if filePathExclusions is not None:
        with open(filePathExclusions):
            for line in swearWords.readlines():
                swearWordsFull = swearWordsFull.replace(line, "")
    if swearWordsFull != "":
        memDB.newDBEntry(type="swearwords", identifier="all", content=swearWordsFull)

initSysPrompts(filePath="insert path to system prompts")
initPersonalityPrompts(filePath="insert path to personality prompts")
initInformationMemory(filePath="insert path to information file")
initSwearWords(filePath="insert path to swear word file")initSysPrompts(filePath="../memories/systemPrompts.txt")

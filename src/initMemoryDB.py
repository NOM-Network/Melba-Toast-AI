import memoryDB
from typing import List

memDB = memoryDB.MemoryDB(path="")

def initSysPrompts(filePath: str) -> bool:
    systemPrompts: List[str] = []
    curPrompt = ""

    with open(filePath) as sysPromptFile:
        for line in sysPromptFile.readlines():
            if line.find("-=sysPromptSplitter=-") != -1:
                systemPrompts.append(curPrompt.replace("\n", ""))
                curPrompt = ""
            else:
                curPrompt += line
        if curPrompt != '':
            systemPrompts.append(curPrompt.replace("\n", ""))

    for prompt in systemPrompts:
        memDB.newDBEntry(type="systemPrompt", identifier="generic", content=prompt)

    return True

def initCharacterMemory(filePath: str) -> bool:
    characterInformation: List[str] = []
    characterNames: List[str] = []
    curInformation = ""
    curCharacter = ""

    with open(filePath) as charInformationFile:
        for line in charInformationFile.readlines():
            if line.find("-=charInfoSplitter=-") != -1:
                characterInformation.append(curInformation.replace("\n", ""))
                characterNames.append(curCharacter.replace("\n", ""))
                curInformation = ""
                curCharacter = ""
            elif line.find("-=charInfoStart=-") != -1:
                curCharacter = line[17:]
            else:
                curInformation += line
        if curInformation != '':
            characterInformation.append(curInformation.replace("\n", ""))
            characterNames.append(curCharacter.replace("\n", ""))
    for name in characterNames:
        memDB.newDBEntry(type="characterinformation", identifier=name,
                         content=characterInformation[characterNames.index(name)])

    return True

success = initSysPrompts(filePath="")
success2 = initCharacterMemory(filePath="")

if success and success2:
    print("Initialized MemoryDB.")
else:
    print("Failed to initialize MemoryDB.")

import melbaToast
melba = melbaToast.Melba("Model Path",
                         "System Prompt Path",
                         "Database Path",
                         "Backup Path") # Backup Path is optional

input = input()
response = melba.getMelbaResponse(input, 0, "username") # message - syspromptsetting - username
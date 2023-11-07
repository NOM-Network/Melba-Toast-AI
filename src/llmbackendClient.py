import sys
sys.path.append("")

from melbaToast import Melba

import json, traceback, time
import websockets.sync.client

wss = True
host = ""
port = 0
endpoint = ""

melba = Melba("openhermes-2-mistral-7b.Q6_K.gguf", "db")

async def handler(request):
    fName = request["fName"].lower()

    if fName == "getmelbaresponse":
        res = melba.getMelbaResponse(message=request["message"],
                                     sysPromptSetting=request["sysPromptSetting"],
                                     person=request["person"])
        if res:
            print("getMelbaResponse: Successfully sent response")
            return {'response' : 'success', 'llmResponse' : res}
        else:
            print("getMelbaResponse: No response from LLM")
    elif fName == "setbackup":
        try:
            melba.setBackup(mode=request["setting"])
            print(f'setBackup: Successfully set backup to {request["setting"]}')
            return {'response' : 'success', 'llmResponse' : f'backup mode: {request["setting"]}'}
        except:
            raise Exception(f'Failed to set backup to {request["setting"]}')
    elif fName == "getemotion":
        res = melba.getEmotion()
        if res:
            print("getEmotion: Successfully sent response")
            return {'response' : 'success', 'llmResponse' : melba.getEmotion()}
        else:
            print("getEmotion: No response from LLM")
    elif fName == "updatellmconfig":
        try:
            melba.updateLLMConfig(newConfig=request["config"])
            print("updateLLMConfig: Successfully updated LLM config")
            return {'response' : 'success', 'llmResponse' : 'updated llmconfig'}
        except:
            print("updateLLMConfig: Filed to update LLM config")
    else:
        print(f"No function with name '{fName}' found")

    return {'response' : 'fail', 'llmResponse' : 'None'}

if __name__ == "__main__":
    while True:
        connection_url = f"{'wss' if wss else 'ws'}://{host}:{port}{endpoint}"
        print(f"Connecting to backend at {connection_url}")
        try:
            with websockets.sync.client.connect(connection_url) as websocket:
                while True:
                    request = json.loads(websocket.recv())
                    print(request)
                    response = handler(request)
                    websocket.send(json.dumps(response))
        except Exception as e:
            print("Exception during WebSocket connection:")
            print(traceback.format_exc())
        print(f"Connection to backend lost, retrying in 3s")
        time.sleep(3)
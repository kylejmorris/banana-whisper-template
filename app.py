from potassium import Potassium, Request, Response
from transformers import pipeline
import whisper
import torch
import os
import base64
from io import BytesIO

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
   
    model = whisper.load_model("base")

    context = {
        "model": model
    }

    return context
    

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    # Parse out your arguments
    print("in handler function")
    mp3BytesString = request.json.get('mp3BytesString', None)
    model = context.get("model")
    print("fetched model and mp3")
    if mp3BytesString == None:
        return Response(
            json = {"error": "No mp3BytesString provided"}, 
            status=500
        )
    
    mp3Bytes = BytesIO(base64.b64decode(mp3BytesString.encode("ISO-8859-1")))
    print("mp3 bytes decoded")
    with open('input.mp3','wb') as file:
        file.write(mp3Bytes.getbuffer())
    
    # Run the model
    print("about to run model")
    result = model.transcribe("input.mp3")
    print("after fetching result of inference")
    output = {"text":result["text"]}
    os.remove("input.mp3")

    return Response(
        json = {"outputs": output}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()
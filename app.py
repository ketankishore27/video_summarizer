import os
from pytube import YouTube
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from flask import Flask, request, jsonify

def audio_downloader(link, destination):
    try:
        object = YouTube(link)
        name = object.streams[0].title
        mins = round(object.length / 60, 2)
        video = YouTube(link)
        audio = video.streams.filter(only_audio=True, file_extension='mp4').first()
        audio.download(filename="sample.mp4")
        return name, mins
    except Exception as e:
        print("Connection Error")
        print(e)
        return False
    
device = "cuda"
context_length = 8000
split_identifier = '[/INST]'
model_name = "Mistral-7B-Instruct-v0.1-GPTQ"
model_llm = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

pipe = pipeline(
    "text-generation",
    model=model_llm,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

prompt = """
Following is the text.
Summarize the text in less than 10 points.
Also be more informative
{}
"""

model_whisper = whisper.load_model("large")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    response_body = {}
    link = request.data.decode('utf-8')
    try:
        file_name = "sample.mp4"
        destination = os.getcwd()
        name, mins = audio_downloader(link, destination)
        result = model_whisper.transcribe(file_name)
        prompt_refined = prompt.format(result['text'][:context_length])
        prompt_template=f'''<s>[INST] {prompt_refined} [/INST]'''
        text_generated = pipe(prompt_template)[0]['generated_text'].split(split_identifier)[-1]
        os.system("rm sample.mp4")
        response_body['video_name'] = name
        response_body['video_duration'] = "{} mins".format(mins)
        response_body['summary'] = text_generated
        response_body['exception'] = None
        return jsonify(response_body)
    except Exception as e:
        response_body['execution_trial'] = 'error'
        response_body['generated_response'] = None
        response_body['exception'] = str(e)
        return jsonify(response_body)
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

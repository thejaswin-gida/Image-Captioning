from openai import OpenAI
client = OpenAI()

import vertexai
from vertexai.vision_models import ImageTextModel, Image
from google.cloud import vision
import os
import tempfile
import gradio as gr
from cryptography.fernet import Fernet
import concurrent.futures
import proto
import json
from deep_translator import GoogleTranslator
from vertexai.preview.language_models import TextGenerationModel
from jose import JWTError, jwt
from dotenv import load_dotenv
import time
from transformers import BlipProcessor, BlipForConditionalGeneration
import io
import boto3
from PIL import Image as PIM
import requests
import warnings
import base64
warnings.filterwarnings("ignore")
# alt,desc,lang,obs,cap,img,llm = "","","","","","",""
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import concurrent.futures
import uuid

language_codes1 = {"afrikaans": "af","albanian": "sq","amharic": "am","arabic": "ar","armenian": "hy","assamese": "as","aymara": "ay","azerbaijani": "az","bambara": "bm","basque": "eu","belarusian": "be","bengali": "bn","bhojpuri": "bho","bosnian": "bs","bulgarian": "bg","catalan": "ca","cebuano": "ceb","chichewa": "ny","chinese (simplified)": "zh-CN","chinese (traditional)": "zh-TW","corsican": "co","croatian": "hr","czech": "cs","danish": "da","dhivehi": "dv","dogri": "doi","dutch": "nl","english": "en","esperanto": "eo","estonian": "et","ewe": "ee","filipino": "tl","finnish": "fi","french": "fr","frisian": "fy","galician": "gl","georgian": "ka","german": "de","greek": "el","guarani": "gn","gujarati": "gu","haitian creole": "ht","hausa": "ha","hawaiian": "haw","hebrew": "iw","hindi": "hi","hmong": "hmn","hungarian": "hu","icelandic": "is","igbo": "ig","ilocano": "ilo","indonesian": "id","irish": "ga","italian": "it","japanese": "ja","javanese": "jw","kannada": "kn","kazakh": "kk","khmer": "km","kinyarwanda": "rw","konkani": "gom","korean": "ko","krio": "kri","kurdish (kurmanji)": "ku","kurdish (sorani)": "ckb","kyrgyz": "ky","lao": "lo","latin": "la","latvian": "lv","lingala": "ln","lithuanian": "lt","luganda": "lg","luxembourgish": "lb","macedonian": "mk","maithili": "mai","malagasy": "mg","malay": "ms","malayalam": "ml","maltese": "mt","maori": "mi","marathi": "mr","meiteilon (manipuri)": "mni-Mtei","mizo": "lus","mongolian": "mn","myanmar": "my","nepali": "ne","norwegian": "no","odia (oriya)": "or","oromo": "om","pashto": "ps","persian": "fa","polish": "pl","portuguese": "pt","punjabi": "pa","quechua": "qu","romanian": "ro","russian": "ru","samoan": "sm","sanskrit": "sa","scots gaelic": "gd","sepedi": "nso","serbian": "sr","sesotho": "st","shona": "sn","sindhi": "sd","sinhala": "si","slovak": "sk","slovenian": "sl","somali": "so","spanish": "es","sundanese": "su","swahili": "sw","swedish": "sv","tajik": "tg","tamil": "ta","tatar": "tt","telugu": "te","thai": "th","tigrinya": "ti","tsonga": "ts","turkish": "tr","turkmen": "tk","twi": "ak","ukrainian": "uk","urdu": "ur","uyghur": "ug","uzbek": "uz","vietnamese": "vi","welsh": "cy","xhosa": "xh","yiddish": "yi","yoruba": "yo","zulu": "zu"}
language_codes2 = {"Afrikaans":"af","Albanian":"sq","Amharic":"am","Arabic":"ar","Armenian":"hy","Azerbaijani":"az","Bengali":"bn","Bosnian":"bs","Bulgarian":"bg","Catalan":"ca","Chinese (Simplified)":"zh","Chinese (Traditional)":"zh-TW","Croatian":"hr","Czech":"cs","Danish":"da","Dari":"fa-AF","Dutch":"nl","English":"en","Estonian":"et","Farsi (Persian)":"fa","Filipino, Tagalog":"tl","Finnish":"fi","French":"fr","French (Canada)":"fr-CA","Georgian":"ka","German":"de","Greek":"el","Gujarati":"gu","Haitian Creole":"ht","Hausa":"ha","Hebrew":"he","Hindi":"hi","Hungarian":"hu","Icelandic":"is","Indonesian":"id","Irish":"ga","Italian":"it","Japanese":"ja","Kannada":"kn","Kazakh":"kk","Korean":"ko","Latvian":"lv","Lithuanian":"lt","Macedonian":"mk","Malay":"ms","Malayalam":"ml","Maltese":"mt","Marathi":"mr","Mongolian":"mn","Norwegian (Bokmål)":"no","Pashto":"ps","Polish":"pl","Portuguese (Brazil)":"pt","Portuguese (Portugal)":"pt-PT","Punjabi":"pa","Romanian":"ro","Russian":"ru","Serbian":"sr","Sinhala":"si","Slovak":"sk","Slovenian":"sl","Somali":"so","Spanish":"es","Spanish (Mexico)":"es-MX","Swahili":"sw","Swedish":"sv","Tamil":"ta","Telugu":"te","Thai":"th","Turkish":"tr","Ukrainian":"uk","Urdu":"ur","Uzbek":"uz","Vietnamese":"vi","Welsh":"cy"}
language_codes3 = {"Afrikaans":"af","Albanian":"sq","Amharic":"am","Arabic":"ar","Armenian":"hy","Assamese":"as","Azerbaijani":"az","Bangla":"bn","Bashkir":"ba","Basque":"eu","Bhojpuri":"bho","Bodo":"brx","Bosnian":"bs","Bulgarian":"bg","Cantonese (Traditional)":"yue","Catalan":"ca","Chinese (Literary)":"lzh","Chinese Simplified":"zh-Hans","Chinese Traditional":"zh-Hant","chiShona":"sn","Croatian":"hr","Czech":"cs","Danish":"da","Dari":"prs","Divehi":"dv","Dogri":"doi","Dutch":"nl","English":"en","Estonian":"et","Faroese":"fo","Fijian":"fj","Filipino":"fil","Finnish":"fi","French":"fr","French (Canada)":"fr-ca","Galician":"gl","Georgian":"ka","German":"de","Greek":"el","Gujarati":"gu","Haitian Creole":"ht","Hausa":"ha","Hebrew":"he","Hindi":"hi","Hmong Daw":"mww","Hungarian":"hu","Icelandic":"is","Igbo":"ig","Indonesian":"id","Inuinnaqtun":"ikt","Inuktitut":"iu","Inuktitut (Latin)":"iu-Latn","Irish":"ga","Italian":"it","Japanese":"ja","Kannada":"kn","Kashmiri":"ks","Kazakh":"kk","Khmer":"km","Kinyarwanda":"rw","Klingon":"tlh-Latn","Klingon (plqaD)":"tlh-Piqd","Konkani":"gom","Korean":"ko","Kurdish (Central)":"ku","Kurdish (Northern)":"kmr","Kyrgyz (Cyrillic)":"ky","Lao":"lo","Latvian":"lv","Lithuanian":"lt","Lingala":"ln","Lower Sorbian":"dsb","Luganda":"lug","Macedonian":"mk","Maithili":"mai","Malagasy":"mg","Malay":"ms","Malayalam":"ml","Maltese":"mt","Maori":"mi","Marathi":"mr","Mongolian (Cyrillic)":"mn-Cyrl","Mongolian (Traditional)":"mn-Mong","Myanmar":"my","Nepali":"ne","Norwegian":"nb","Nyanja":"nya","Odia":"or","Pashto":"ps","Persian":"fa","Polish":"pl","Portuguese (Brazil)":"pt","Portuguese (Portugal)":"pt-pt","Punjabi":"pa","Queretaro Otomi":"otq","Romanian":"ro","Rundi":"run","Russian":"ru","Samoan (Latin)":"sm","Serbian (Cyrillic)":"sr-Cyrl","Serbian (Latin)":"sr-Latn","Sesotho":"st","Sesotho sa Leboa":"nso","Setswana":"tn","Sindhi":"sd","Sinhala":"si","Slovak":"sk","Slovenian":"sl","Somali (Arabic)":"so","Spanish":"es","Swahili (Latin)":"sw","Swedish":"sv","Tahitian":"ty","Tamil":"ta","Tatar":"tt","Telugu":"te","Thai":"th","Tibetan":"bo","Tigrinya":"ti","Tongan":"to","Turkish":"tr","Turkmen":"tk","Ukrainian":"uk","Upper Sorbian":"hsb","Urdu":"ur","Uyghur (Arabic)":"ug","Uzbek":"uz","Vietnamese":"vi","Welsh":"cy","Xhosa":"xh","Yoruba":"yo","Yucatec Maya":"yua","Zulu":"zu"}
language_codes4 = {"Hindi":"hi","Punjabi":"pa","Tamil":"ta","Gujarati":"gu","Kannada":"kn","Bengali":"bn","Marathi":"mr","Telugu":"te","English":"en","Malayalam":"ml","Assamese":"as","Odia":"or","French":"fr","Arabic":"ar","German":"de","Spanish":"es","Japanese":"ja","Italian":"it","Dutch":"nl","Portuguese":"pt","Vietnamese":"vi","Indonesian":"id","Urdu":"ur","Chinese (Simplified)":"zh-CN","Chinese (Traditional)":"zh-TW","Kashmiri":"ksm","Konkani":"gom","Manipuri":"mni-Mtei","Nepali":"ne","Sanskrit":"sa","Sindhi":"sd","Bodo":"bodo","Santhali":"snthl","Maithili":"mai","Dogri":"doi","Malay":"ms","Filipino":"tl"}

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
encryption_key = os.environ['enc_key']
fernet = Fernet(encryption_key)

rekognition = boto3.client(
    'rekognition',
    region_name='us-east-1',
    aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
)

aws_client = boto3.client('translate',region_name='us-east-1',
    aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
)

BLIPprocessor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
BLIPmodel = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Decrypt the encrypted data
with open("encrypted_data.json", "rb") as encrypted_file:
    encrypted_data = encrypted_file.read()
    decrypted_data = fernet.decrypt(encrypted_data)

with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
    temp_file.write(decrypted_data)
    temp_file_path = temp_file.name

# Set the temporary file path as the Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path

PROJECT_ID = os.environ['project_id']
LOCATION = 'us-central1'
vertexai.init(project=PROJECT_ID, location=LOCATION)
VISIONmodel = ImageTextModel.from_pretrained("imagetext@001")
VISIONclient = vision.ImageAnnotatorClient()

# GPT4 to generate descriptions
def GPT(gptmodel, para):
    gptmodel=gptmodel.lower()
    if gptmodel=='gpt-4-turbo':
        gptmodel='gpt-4-1106-preview'
    s = time.time()
    feeder = (
        "Generate a comprehensive description of the image's content for its alt text. Pay attention to available details and emotions conveyed. Avoid personal analysis and keep the description concise.. The details are as follows - \n"
        + para
    )
    formatter = "Craft a coherent image description that encompasses all available information. Prioritize accuracy and coherence without adding extra details. Be creative while maintaining simplicity. Aim for a description of around 100 words without including additional information."
    curator = "When describing a poster, exercise caution, particularly when posters convey deep emotions. Differentiate between objects and the conveyed emotions. Avoid redundant phrases like 'This message appears multiple times' and concentrate on capturing the essence of the content and emotions conveyed."

    instruction = "start with 'A picture of '"

    conversation = [
        {"role": "system", "content": "You are an AI language model."},
        {"role": "user", "content": feeder},
        {"role": "user", "content": formatter},
        {"role": "user", "content": curator},
        {"role": "user", "content": instruction},
    ]

    response = client.chat.completions.create(
        model=gptmodel,
        messages=conversation,
    )

    # print("desc - ", time.time() - s)
    return response.choices[0].message.content

# PaLM2 to generate descriptions
def PaLM2des(
    project_id: str,
    model_name: str,
    temperature: float,
    max_decode_steps: int,
    top_p: float,
    top_k: int,
    content: str,
    location: str = "us-central1",
    tuned_model_name: str = "",
):
    infoString = [
        {"role": "system", "content": "You are an AI language model."},
        {
            "role": "user",
            "content": "Generate a comprehensive description of the image's content for its alt text. Pay attention to available details and emotions conveyed. Avoid personal analysis and keep the description concise.. The details are as follows - \n"
            + content,
        },
        {
            "role": "user",
            "content": "Craft a coherent image description that encompasses all available information. Prioritize accuracy and coherence without adding extra details. Be creative while maintaining simplicity. Aim for a description of around 100 words without including additional information.",
        },
        {
            "role": "user",
            "content": "When describing a poster, exercise caution, particularly when posters convey deep emotions. Differentiate between objects and the conveyed emotions. Avoid redundant phrases like 'This message appears multiple times' and concentrate on capturing the essence of the content and emotions conveyed.",
        },
        {"role": "user", "content": "start with 'A picture of '"},
    ]

    content = "follow the instructions in this conversation structure below\n" + str(
        infoString
    )

    model = TextGenerationModel.from_pretrained(model_name)

    if tuned_model_name:
        model = model.get_tuned_model(tuned_model_name)

    response = model.predict(
        content,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
        top_k=top_k,
        top_p=top_p,
    )

    return response.text

# Llama2 to generate descriptions
def Llama2des(text):
    pplx_api_key=os.environ['pplx_api_key']
    payload = {
        "model": "llama-2-70b-chat",
        "messages": [
            {"role": "system", "content": "You are an AI language model."},
            {
                "role": "user",
                "content": "Generate a comprehensive description of the image's content for its alt text. Pay attention to available details and emotions conveyed. Avoid personal analysis and keep the description concise.. The details are as follows - \n"
                + text,
            },
            {
                "role": "user",
                "content": "Craft a coherent image description that encompasses all available information. Prioritize accuracy and coherence without adding extra details. Be creative while maintaining simplicity. Aim for a description of around 100 words without including additional information.",
            },
            {
                "role": "user",
                "content": "When describing a poster, exercise caution, particularly when posters convey deep emotions. Differentiate between objects and the conveyed emotions. Avoid redundant phrases like 'This message appears multiple times' and concentrate on capturing the essence of the content and emotions conveyed.",
            },
            {"role": "user", "content": "Strictly start with 'A picture of '"},
        ],
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {pplx_api_key}",
    }

    response = requests.post(
        "https://api.perplexity.ai/chat/completions", json=payload, headers=headers
    ).json()

    return response["choices"][0]["message"]["content"]

def gpt4V(path):
    key=os.environ['OPENAI_API_KEY']
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Getting the base64 string
    base64_image = encode_image(path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Write a caption and a 100-word description for this image by trying to answer the following questions:
                1. Who are the people in this picture?
                2. What objects or subjects does this image contain?
                3. Extract all the text and numbers from this image.
                4. Describe the events or activities happening in this image.
                5. If applicable, describe the emotions portrayed in the image.

                ** If any specific information is missing, please naturally incorporate the available details into your response. Avoid stating the absence of certain elements explicitly; instead, prioritize creating a comprehensive and engaging narrative around the visible content. Ensure that the description is grammatically correct and coherent.                
                If specific details are not visible/not depicted, seamlessly integrate the available information into your narrative without explicitly stating the absence of certain elements.
                Strictly Avoid saying "There are No x, y, z present/depicited in image." **
                
                Please provide a JSON response in the following format - 
                {
                    "altText": "Enter a suitable title or alt text for the image",
                    "Description": "Compose a detailed description of the image content with a 100-word limit."
                }
                In your response, feel free to omit any information that is not applicable or missing from the image.""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 4096,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    ).json()
    # print(response)

    string = response["choices"][0]["message"]["content"]
    # print(string)
    a=json.loads("{" + str(string.split("{")[-1].split("}")[0]) + "}")
    return a['altText'],a['Description']

# check for obscenity in images and flag the ones which are NSFW
def gcVision(content, adult, spoof, medical, violence, racy): 
    s = time.time()
    image = vision.Image(content=content)
    response = VISIONclient.safe_search_detection(image=image)
    safe = response.safe_search_annotation
    safe = proto.Message.to_json(safe)
    safe = json.loads(safe)
    print(safe)
    print(adult)
    print(spoof)
    print(medical)
    print(violence)
    print(racy)
    print([safe[key] > value for key, value in {'adult': adult, 'spoof': spoof, 'medical': medical, 'violence': violence, 'racy': racy}.items()])
    if any(safe[key] > value for key, value in {'adult': adult, 'spoof': spoof, 'medical': medical, 'violence': violence, 'racy': racy}.items()):
        return True
    return False


# AWS Rekognition module for obscenity check
def AWSrekognition(path, explicit_nudity, suggestive, violence, visually_disturbing, rude_gestures, drugs, tobacco, alcohol, gambling, hate_symbols):
    s = time.time()
    # Read the image file as bytes
    with open(path, "rb") as image_file:
        image_bytes = image_file.read()
    try:
        response = rekognition.detect_moderation_labels(
            Image={"Bytes": image_bytes}, MinConfidence=0
        )
        print(response)
        print("rekognition - ", time.time() - s)

        # Create a dictionary to store the highest confidence for each top-level category
        top_level_confidences = {
            "Explicit Nudity": 0,
            "Suggestive": 0,
            "Violence": 0,
            "Visually Disturbing": 0,
            "Rude Gestures": 0,
            "Drugs": 0,
            "Tobacco": 0,
            "Alcohol": 0,
            "Gambling": 0,
            "Hate Symbols": 0,
        }

        # Iterate through the ModerationLabels and update the highest confidence for each top-level category
        for label in response["ModerationLabels"]:
            name = label["Name"]
            confidence = label["Confidence"]

            if name in top_level_confidences.keys():
                top_level_confidences[name] = confidence

        # Compare the highest confidence for each top-level category with the function parameters
        if any(top_level_confidences[key] > value for key, value in {
            "Explicit Nudity": explicit_nudity,
            "Suggestive": suggestive,
            "Violence": violence,
            "Visually Disturbing": visually_disturbing,
            "Rude Gestures": rude_gestures,
            "Drugs": drugs,
            "Tobacco": tobacco,
            "Alcohol": alcohol,
            "Gambling": gambling,
            "Hate Symbols": hate_symbols,
        }.items()):
            return True
        else:
            return False

    except Exception as e:
        print(e)

# ask questions on the image and return one answer at a time
def get_answers(image, question):
    s = time.time()
    answers = VISIONmodel.ask_question(
        image=image, question=question, number_of_results=3
    )  # Get 3 results
    valid_answers = [
        ans
        for ans in answers
        if ans
        not in (
            "unanswered",
            "no text in image",
            "no",
            "unanswerable",
            "no text",
            "unsuitable",
        )
    ]
    if valid_answers:
        # print("getans - ", time.time() - s)
        return " ".join(list(set(valid_answers)))
    return []

# function to concurrently generate answers based on questions asked on images
def collect_answers(image, questions):
    s = time.time()
    all_answers = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(get_answers, image, question) for question in questions
        ]
        for future in concurrent.futures.as_completed(futures):
            ans = future.result()
            # print(ans)
            if ans:
                all_answers.append(ans)  # Add answers to the combined list
    # print("collectans - ", time.time() - s)
    # print(all_answers)
    return "".join(all_answers)

# Generating captions with google cloud vision
def VISIONcap(image_data, desModel):
    s = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image_file:
        temp_image_path = temp_image_file.name
        temp_image_file.write(image_data)

    try:
        image = Image.load_from_file(location=temp_image_path)
        questions = [
            "name the people in this picture.",
            "What does this image contain?",
            "Give me all the text and all the numbers in this image.",
            "What is happening in this image?",
            "what are the emotions being potrayed in the image? do not answer if not necessary.",
        ]

        # Create a ThreadPoolExecutor for concurrent execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the collect_answers function to the executor
            answers_future = executor.submit(collect_answers, image, questions)

            # Use the model.get_captions function directly
            captions = VISIONmodel.get_captions(
                image=image, number_of_results=1, language="en"
            )

        # Get the results from the answers and captions futures
        answers = answers_future.result()
        # print(answers)
        alt = captions[0].capitalize()
        desString = (
            "\nAlt Text of the image - "
            + alt
            + "\nObjects and emotions in the picture - "
            + answers
        )
        des = genDescription(desString, desModel)
        # print(alt, des)
        # print("generate - ", time.time() - s)

        return alt, des

    except Exception as e:
        return e, e


def split_text(input_text, max_chunk_length=4995):
    sentences = sent_tokenize(input_text)

    # Initialize variables
    chunks = []
    current_chunk = ""

    # Iterate through sentences
    for sentence in sentences:
        # Check if adding the sentence to the current chunk exceeds the max length
        if len(current_chunk) + len(sentence) <= max_chunk_length:
            current_chunk += sentence + " "
        else:
            # Save the current chunk and start a new one with the current sentence
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def gtranslate(language, text):
    return GoogleTranslator(source="auto", target=language_codes1[language]).translate(text)

def google_translate(text, language):
    if len(text)>=5000:
        print('Character length greater than 5000')
        result_chunks = split_text(text)
        language = [language]*len(result_chunks)
        with concurrent.futures.ThreadPoolExecutor() as executor:
          result_chunks = list(executor.map(gtranslate, language, result_chunks))
        return ' '.join(result_chunks)
    return GoogleTranslator(source="auto", target=language_codes1[language]).translate(text)

def aws_translate(language,text):
    response = aws_client.translate_text(
    Text=text,
    SourceLanguageCode='auto',
    TargetLanguageCode=language_codes2[language],
    )
    return response['TranslatedText']

def azure_translate(language,text):
    key = os.environ['azure_api_key']
    endpoint = "https://api.cognitive.microsofttranslator.com"
    location = "eastus"
    
    path = '/translate'
    constructed_url = endpoint + path
    
    params = {
        'api-version': '3.0',
        'to': [language_codes3[language]]
    }
    
    headers = {
        'Ocp-Apim-Subscription-Key': key,
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    
    # You can pass more than one object in body.
    body = [{'text': text}]
    
    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()
    return response[0]["translations"][0]['text']

def devnagri_translate(language, text):
    url = "https://app.devnagri.com/api/translate-sentence-api"
    # result = detect(text=text, low_memory=False)
    payload = {
        "key": os.environ['devnagri_api_key'],
        "sentence": text,
        "src_lang": "en",
        "dest_lang": language_codes4[language],
    }

    response = requests.post(url, data=payload)

    try:
        return json.loads(response.text)["translated_text"]
    except Exception as e:
        print(e)
        return "Couldn't provide response for the desired language because of devnagri.com."    

def translate(language_provider, language, text):
    if language_provider=='Google Translate':
        return google_translate(language, text)
    elif language_provider == 'AWS Translate':
        return aws_translate(language, text)
    elif language_provider == 'Azure AI Translator':
        return azure_translate(language, text)
    else:
        return devnagri_translate(language, text)

# Generation Captions with salesforce BLIP
def BLIPcaptions(raw_image, desModel):
    # conditional image captioning
    text = "A picture of"
    inputs = BLIPprocessor(
        raw_image, text, return_tensors="pt"
    )  # .to("cuda", torch.float16)

    out = BLIPmodel.generate(**inputs)

    return BLIPprocessor.decode(out[0], skip_special_tokens=True), genDescription(
        BLIPprocessor.decode(out[0], skip_special_tokens=True), desModel
    )

# Flow control function
def genDescription(infoString, desModel):
    if desModel == "Llama2":
        return Llama2des(infoString)
    elif desModel[:3] == "GPT":
        return GPT(desModel,infoString)
    elif desModel == "PaLM2":
        return PaLM2des("data-science-401804","text-bison",0.2,512,0.8,40,infoString,"us-central1",)
    else:
        return "Unrecognized Description Provider"

def predict(image, obs_model, slider1, slider2, slider3, slider4, slider5, slider6, slider7, slider8, slider9, slider10, img_model, desc_model, language_provider, language):
    print(image)
    print(obs_model)
    print(img_model)
    print(desc_model)
    print(language_provider)
    print(language)

    obs_model = obs_model.split()[0]
    img_model = img_model.split()[0]
    desc_model = desc_model.split()[0]
    language_provider = language_provider.split('(')[0].strip()

    with open(image, 'rb') as image_file:
        image_bytes = image_file.read()
    img = PIM.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")

    # Save the converted image back to bytes
    img_byte_array = io.BytesIO()
    img.save(img_byte_array, format="PNG")

    # Use the `getvalue()` method to retrieve the bytes-like object
    img_bytes = img_byte_array.getvalue()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image_file:
        temp_image_path = temp_image_file.name
        temp_image_file.write(img_bytes)

    # Obscenity Providers
    if obs_model == "awsRekognition":
        if AWSrekognition(temp_image_path, slider1, slider2, slider3, slider4, slider5, slider6, slider7, slider8, slider9, slider10):
            return "Unsafe Image!", "Try again with a different Image!"
    elif obs_model == "gcVision":
        if gcVision(img_bytes, slider1, slider2, slider3, slider4, slider5):
            return "Unsafe Image!", "Try again with a different Image!"
    else:
        return "Incorrect Provider for obscenity check!", "Choose one of awsRekognition and gcVision!"

    # Caption Providers
    if img_model == "gcVertex":
        alt, des = VISIONcap(img_byte_array.getvalue(), desc_model)
    elif img_model == "sfBLIP":
        alt, des = BLIPcaptions(PIM.open(temp_image_path), desc_model)
    elif img_model == "gpt4V":
        alt, des = gpt4V(temp_image_path)
    else:
        return "Incorrect Provider for generating content!", "Choose one of gcVertex and sfBLIP!"


    # Language Translation
    if not language:
        print(alt)
        print(des)
        return alt, des

    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            alt_future = executor.submit(translate,language_provider ,alt ,language)
            des_future = executor.submit(translate,language_provider, des , language)
        alt=alt_future.result()
        des=des_future.result()
        print(alt)
        print(des)
        return alt, des

model_dropdown_choices = [
    "GPT-3.5-turbo ($0.0002 + $0.0002 / 100 output words)",
    "GPT-4-turbo ($0.002 + $0.004 / 100 output words)",
    "GPT-4 ($0.006 + $0.008 / 100 output words)",
    "PALM2 ($0.0001 + $0.00006 / 100 output words)",
    "Llama2 ($0.001 / requests)"
]

# Language choices based on language provider
languages = ['Google Translate (Free)', 'AWS Translate ($0.015/ 1K characters)', 'Azure AI Translator ($0.01 / 1K Characters)', 'Devnagri.com (Rs.1 / 1K characters)']
language_choices_map = {
    "Google Translate (Free)": ['afrikaans', 'albanian', 'amharic', 'arabic', 'armenian', 'assamese', 'aymara', 'azerbaijani', 'bambara', 'basque', 'belarusian', 'bengali', 'bhojpuri', 'bosnian', 'bulgarian', 'catalan', 'cebuano', 'chichewa', 'chinese (simplified)', 'chinese (traditional)', 'corsican', 'croatian', 'czech', 'danish', 'dhivehi', 'dogri', 'dutch', 'english', 'esperanto', 'estonian', 'ewe', 'filipino', 'finnish', 'french', 'frisian', 'galician', 'georgian', 'german', 'greek', 'guarani', 'gujarati', 'haitian creole', 'hausa', 'hawaiian', 'hebrew', 'hindi', 'hmong', 'hungarian', 'icelandic', 'igbo', 'ilocano', 'indonesian', 'irish', 'italian', 'japanese', 'javanese', 'kannada', 'kazakh', 'khmer', 'kinyarwanda', 'konkani', 'korean', 'krio', 'kurdish (kurmanji)', 'kurdish (sorani)', 'kyrgyz', 'lao', 'latin', 'latvian', 'lingala', 'lithuanian', 'luganda', 'luxembourgish', 'macedonian', 'maithili', 'malagasy', 'malay', 'malayalam', 'maltese', 'maori', 'marathi', 'meiteilon (manipuri)', 'mizo', 'mongolian', 'myanmar', 'nepali', 'norwegian', 'odia (oriya)', 'oromo', 'pashto', 'persian', 'polish', 'portuguese', 'punjabi', 'quechua', 'romanian', 'russian', 'samoan', 'sanskrit', 'scots gaelic', 'sepedi', 'serbian', 'sesotho', 'shona', 'sindhi', 'sinhala', 'slovak', 'slovenian', 'somali', 'spanish', 'sundanese', 'swahili', 'swedish', 'tajik', 'tamil', 'tatar', 'telugu', 'thai', 'tigrinya', 'tsonga', 'turkish', 'turkmen', 'twi', 'ukrainian', 'urdu', 'uyghur', 'uzbek', 'vietnamese', 'welsh', 'xhosa', 'yiddish', 'yoruba', 'zulu'],
    "AWS Translate ($0.015/ 1K characters)": ['Afrikaans', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Azerbaijani', 'Bengali', 'Bosnian', 'Bulgarian', 'Catalan', 'Chinese (Simplified)', 'Chinese (Traditional)', 'Croatian', 'Czech', 'Danish', 'Dari', 'Dutch', 'English', 'Estonian', 'Farsi (Persian)', 'Filipino, Tagalog', 'Finnish', 'French', 'French (Canada)', 'Georgian', 'German', 'Greek', 'Gujarati', 'Haitian Creole', 'Hausa', 'Hebrew', 'Hindi', 'Hungarian', 'Icelandic', 'Indonesian', 'Irish', 'Italian', 'Japanese', 'Kannada', 'Kazakh', 'Korean', 'Latvian', 'Lithuanian', 'Macedonian', 'Malay', 'Malayalam', 'Maltese', 'Marathi', 'Mongolian', 'Norwegian (Bokmål)', 'Pashto', 'Polish', 'Portuguese (Brazil)', 'Portuguese (Portugal)', 'Punjabi', 'Romanian', 'Russian', 'Serbian', 'Sinhala', 'Slovak', 'Slovenian', 'Somali', 'Spanish', 'Spanish (Mexico)', 'Swahili', 'Swedish', 'Tamil', 'Telugu', 'Thai', 'Turkish', 'Ukrainian', 'Urdu', 'Uzbek', 'Vietnamese', 'Welsh'],
    "Azure AI Translator ($0.01 / 1K Characters)": ['Afrikaans', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Assamese', 'Azerbaijani', 'Bangla', 'Bashkir', 'Basque', 'Bhojpuri', 'Bodo', 'Bosnian', 'Bulgarian', 'Cantonese (Traditional)', 'Catalan', 'Chinese (Literary)', 'Chinese Simplified', 'Chinese Traditional', 'chiShona', 'Croatian', 'Czech', 'Danish', 'Dari', 'Divehi', 'Dogri', 'Dutch', 'English', 'Estonian', 'Faroese', 'Fijian', 'Filipino', 'Finnish', 'French', 'French (Canada)', 'Galician', 'Georgian', 'German', 'Greek', 'Gujarati', 'Haitian Creole', 'Hausa', 'Hebrew', 'Hindi', 'Hmong Daw', 'Hungarian', 'Icelandic', 'Igbo', 'Indonesian', 'Inuinnaqtun', 'Inuktitut', 'Inuktitut (Latin)', 'Irish', 'Italian', 'Japanese', 'Kannada', 'Kashmiri', 'Kazakh', 'Khmer', 'Kinyarwanda', 'Klingon', 'Klingon (plqaD)', 'Konkani', 'Korean', 'Kurdish (Central)', 'Kurdish (Northern)', 'Kyrgyz (Cyrillic)', 'Lao', 'Latvian', 'Lithuanian', 'Lingala', 'Lower Sorbian', 'Luganda', 'Macedonian', 'Maithili', 'Malagasy', 'Malay', 'Malayalam', 'Maltese', 'Maori', 'Marathi', 'Mongolian (Cyrillic)', 'Mongolian (Traditional)', 'Myanmar', 'Nepali', 'Norwegian', 'Nyanja', 'Odia', 'Pashto', 'Persian', 'Polish', 'Portuguese (Brazil)', 'Portuguese (Portugal)', 'Punjabi', 'Queretaro Otomi', 'Romanian', 'Rundi', 'Russian', 'Samoan (Latin)', 'Serbian (Cyrillic)', 'Serbian (Latin)', 'Sesotho', 'Sesotho sa Leboa', 'Setswana', 'Sindhi', 'Sinhala', 'Slovak', 'Slovenian', 'Somali (Arabic)', 'Spanish', 'Swahili (Latin)', 'Swedish', 'Tahitian', 'Tamil', 'Tatar', 'Telugu', 'Thai', 'Tibetan', 'Tigrinya', 'Tongan', 'Turkish', 'Turkmen', 'Ukrainian', 'Upper Sorbian', 'Urdu', 'Uyghur (Arabic)', 'Uzbek', 'Vietnamese', 'Welsh', 'Xhosa', 'Yoruba', 'Yucatec Maya', 'Zulu'],
    "Devnagri.com (Rs.1 / 1K characters)": ['Hindi', 'Punjabi', 'Tamil', 'Gujarati', 'Kannada', 'Bengali', 'Marathi', 'Telugu', 'English', 'Malayalam', 'Assamese', 'Odia', 'French', 'Arabic', 'German', 'Spanish', 'Japanese', 'Italian', 'Dutch', 'Portuguese', 'Vietnamese', 'Indonesian', 'Urdu', 'Chinese (Simplified)', 'Chinese (Traditional)', 'Kashmiri', 'Konkani', 'Manipuri', 'Nepali', 'Sanskrit', 'Sindhi', 'Bodo', 'Santhali', 'Maithili', 'Dogri', 'Malay', 'Filipino']
}

def rs_change(rs):
    return gr.update(choices=language_choices_map[rs], value=None)

def change_slider1(choice):
    if choice == "gcVision ($0.0015 / image)":
        return gr.Slider(minimum=1, maximum=5, visible=True, step=1, value=3, label="Select the obscenity level for adult content")
    elif choice == "awsRekognition ($0.001/ image)":
        return gr.Slider(minimum=0, maximum=100, visible=True, step=5, value=75, label="Select the obscenity level for Explicit Nudity content")

def change_slider2(choice):
    if choice == "gcVision ($0.0015 / image)":
        return gr.Slider(minimum=1, maximum=5, visible=True, step=1, value=3, label="Select the obscenity level for spoof content")
    elif choice == "awsRekognition ($0.001/ image)":
        return gr.Slider(minimum=0, maximum=100, visible=True, step=5, value=75, label="Select the obscenity level for Suggestive content")

def change_slider3(choice):
    if choice == "gcVision ($0.0015 / image)":
        return gr.Slider(minimum=1, maximum=5, visible=True,step=1, value=3, label="Select the obscenity level for medical content")
    elif choice == "awsRekognition ($0.001/ image)":
        return gr.Slider(minimum=0, maximum=100, visible=True, step=5, value=75, label="Select the obscenity level for Violence content")

def change_slider4(choice):
    if choice == "gcVision ($0.0015 / image)":
        return gr.Slider(minimum=1, maximum=5, visible=True,step=1, value=3, label="Select the obscenity level for violence content")
    elif choice == "awsRekognition ($0.001/ image)":
        return gr.Slider(minimum=0, maximum=100, visible=True, step=5, value=75, label="Select the obscenity level for Visually Disturbing content")

def change_slider5(choice):
    if choice == "gcVision ($0.0015 / image)":
        return gr.Slider(minimum=1, maximum=5, visible=True,step=1, value=3, label="Select the obscenity level for racy content")
    elif choice == "awsRekognition ($0.001/ image)":
        return gr.Slider(minimum=0, maximum=100, visible=True, step=5, value=75, label="Select the obscenity level for Rude Gestures content")

def change_slider6(choice):
    if choice == "gcVision ($0.0015 / image)":
        return gr.Slider(minimum=1, maximum=5, visible=False, step=1, value=3,)
    elif choice == "awsRekognition ($0.001/ image)":
        return gr.Slider(minimum=0, maximum=100, visible=True, step=5, value=75, label="Select the obscenity level for Drugs content")

def change_slider7(choice):
    if choice == "gcVision ($0.0015 / image)":
        return gr.Slider(minimum=1, maximum=5, visible=False, step=1, value=3)
    elif choice == "awsRekognition ($0.001/ image)":
        return gr.Slider(minimum=0, maximum=100, visible=True, step=5, value=75, label="Select the obscenity level for Tobacco content")

def change_slider8(choice):
    if choice == "gcVision ($0.0015 / image)":
        return gr.Slider(minimum=1, maximum=5, visible=False, step=1, value=3)
    elif choice == "awsRekognition ($0.001/ image)":
        return gr.Slider(minimum=0, maximum=100, visible=True, step=5, value=75, label="Select the obscenity level for Alcohol content")

def change_slider9(choice):
    if choice == "gcVision ($0.0015 / image)":
        return gr.Slider(minimum=1, maximum=5, visible=False, step=1,value=3)
    elif choice == "awsRekognition ($0.001/ image)":
        return gr.Slider(minimum=0, maximum=100, visible=True, step=5, value=75, label="Select the obscenity level for Gambling content")

def change_slider10(choice):
    if choice == "gcVision ($0.0015 / image)":
        return gr.Slider(minimum=1, maximum=5, visible=False, step=1,value=3)
    elif choice == "awsRekognition ($0.001/ image)":
        return gr.Slider(minimum=0, maximum=100, visible=True, step=5, value=75, label="Select the obscenity level for Hate Symbols content")

with gr.Blocks() as app:
    image = gr.File(label="Upload Image")
    # obs_model = gr.Dropdown(["awsRekognition ($0.001/ image)", "gcVision ($0.0015 / image)"], label="Select Obscenity Model")
    obs_radio = gr.Radio(
        ["awsRekognition ($0.001/ image)", "gcVision ($0.0015 / image)"], label="Select Obscenity Model"
    )
    cap_model = gr.Dropdown(["gcVertex ($0.0015 / image)","gpt4V ($0.009+ $0.004 / 100 output words)", "sfBLIP (Free)"], label="Select Caption Provider Model")
    model_info = gr.Dropdown(label="Select Text Generation Model", choices=model_dropdown_choices)
    language_provider = gr.Dropdown(label="Language Translation Provider", choices=languages, value='Google Translate (Free)')
    language_dropdown = gr.Dropdown(label="Language", choices=language_choices_map['Google Translate (Free)'], interactive=True)
    language_provider.change(fn=rs_change,  inputs=[language_provider], outputs=[language_dropdown])

    slider1 = gr.Slider(minimum=0, maximum=10, visible=False, label="Choose quantity of")
    slider2 = gr.Slider(minimum=0, maximum=10, visible=False, label="Choose quantity of")
    slider3 = gr.Slider(minimum=0, maximum=10, visible=False, label="Choose quantity of")
    slider4 = gr.Slider(minimum=0, maximum=10, visible=False, label="Choose quantity of")
    slider5 = gr.Slider(minimum=0, maximum=10, visible=False, label="Choose quantity of")
    slider6 = gr.Slider(minimum=0, maximum=10, visible=False, label="Choose quantity of")
    slider7 = gr.Slider(minimum=0, maximum=10, visible=False, label="Choose quantity of")
    slider8 = gr.Slider(minimum=0, maximum=10, visible=False, label="Choose quantity of")
    slider9 = gr.Slider(minimum=0, maximum=10, visible=False, label="Choose quantity of")
    slider10 = gr.Slider(minimum=0, maximum=10, visible=False, label="Choose quantity of")
    obs_radio.change(fn = change_slider1, inputs = obs_radio, outputs = slider1)
    obs_radio.change(fn = change_slider2, inputs = obs_radio, outputs = slider2)
    obs_radio.change(fn = change_slider3, inputs = obs_radio, outputs = slider3)
    obs_radio.change(fn = change_slider4, inputs = obs_radio, outputs = slider4)
    obs_radio.change(fn = change_slider5, inputs = obs_radio, outputs = slider5)
    obs_radio.change(fn = change_slider6, inputs = obs_radio, outputs = slider6)
    obs_radio.change(fn = change_slider7, inputs = obs_radio, outputs = slider7)
    obs_radio.change(fn = change_slider8, inputs = obs_radio, outputs = slider8)
    obs_radio.change(fn = change_slider9, inputs = obs_radio, outputs = slider9)
    obs_radio.change(fn = change_slider10, inputs = obs_radio, outputs = slider10)
    
    gr.Interface(
        fn=predict,
        inputs=[image, obs_radio, slider1, slider2, slider3, slider4, slider5, slider6, slider7, slider8, slider9, slider10, cap_model, model_info, language_provider, language_dropdown],
        outputs=[
            gr.Textbox(label="Alt Text"),
            gr.Textbox(label="Description")
        ],
        title="Alt Text and Description Generator",
        description="Generate alt text and description for an uploaded image using AI models.",
        server_name="0.0.0.0"
    )

app.launch(debug=True, share=True)
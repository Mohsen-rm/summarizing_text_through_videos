import subprocess
import threading
import os
from time import sleep
from pytube import YouTube
from flask import Flask, request, render_template, jsonify
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from googletrans import Translator
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
from nltk.tokenize import word_tokenize
from gtts import gTTS

nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)

# Create a speech recognition object
r = sr.Recognizer()

key_url_array = []
language_arr = ['ar','en']

def delete_files_in_folder(folder_path):
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Iterate over the files and delete each one
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error deleting file: {file_path}, {e}")

#delete file old from folder
delete_files_in_folder("audio-chunks")
delete_files_in_folder("file")
delete_files_in_folder("videos")
delete_files_in_folder("wav")
delete_files_in_folder("static/mp3")

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/getDataTxt', methods=['GET'])
def get_data():
    key = request.args.get('key')
    folder = 'file'
    filename = key + '.txt'
    if check_file_in_folder(folder, filename):
        print(f"File '{filename}' exists in folder '{folder}'.")
        with open('file/'+key+'.txt', 'r') as file:
            text_data = file.read()
        return jsonify({"is_finished": True, "utext": text_data})
    else:
        return jsonify({"is_finished": False,"utext": ""})

@app.route('/Translator', methods=['GET'])
def get_translator():
    words = request.args.get('text')
    try:
        translated_word = translate_to_arabic(words)
        return jsonify({"is_finished": True,"utext": translated_word})
    except:
        return jsonify({"is_finished": False, "utext": ""})

@app.route('/Summarize_text', methods=['GET'])
def get_summarize_text():
    words = request.args.get('text')
    try:
        summary = summarize_text(words)
        return jsonify({"is_finished": True,"utext": "The New Text is \n"+summary})
    except:
        return jsonify({"is_finished": False, "utext": ""})

@app.route('/Extract_topics', methods=['GET'])
def get_extract_topics():
    words = request.args.get('text')
    try:
        topics = extract_topics(words)
        return jsonify({"is_finished": True,"utext": "The Topics of Text is \n"+str(topics)})
    except:
        return jsonify({"is_finished": False, "utext": ""})

@app.route('/Text_to_audio', methods=['GET'])
def get_text_to_audio():
    key = request.args.get('key')
    words = request.args.get('text')
    try:
        text_to_audio(key+'_en.mp3',words,"en")
        text_to_audio(key + '_ar.mp3', translate_to_arabic(words), "ar")
        var_en = "http://127.0.0.1:5000/static/mp3/" + key + '_en.mp3'
        var_ar = "http://127.0.0.1:5000/static/mp3/" + key + '_ar.mp3'
        return jsonify({"is_finished": True,"utext":f"<a href='{var_en}' target='_blank'>mp3 EN</a> <br> <a href='{var_ar}'  target='_blank'>mp3 AR</a>"})
    except:
        return jsonify({"is_finished": False, "utext": ""})


@app.route('/', methods=['POST'])
def handle_post_request():
    # Get the data from the POST request
    ukey = request.form.get('ukey')
    url = request.form.get('url')
    key_url_array.append((ukey, url,False))
    return 'OK'

def text_to_audio(path,text, language='en'):
    # Create a gTTS object
    tts = gTTS(text=text, lang=language, slow=False)
    # Save the audio to a file
    tts.save("static/mp3/"+path)

def extract_topics(text, max_topics=5):
    # تقسيم النص إلى كلمات
    tokens = word_tokenize(text)
    # وسم أنواع الكلمات
    tagged_tokens = nltk.pos_tag(tokens)
    # استخراج المواضيع بناءً على أنواع الكلمات
    topics = []
    count = 0
    for word, tag in tagged_tokens:
        # اختر الكلمات ذات الأهمية بناءً على أنواع الكلمات المعنية لك
        if tag.startswith('NN'):  # اختيار الأسماء العامة كمواضيع
            topics.append(word)
            count += 1
            if count == max_topics:
                break  # توقف بعد الوصول إلى الحد الأقصى للمواضيع

    return topics

def summarize_text(text, sentences_count=2):
    # Initialize the parser with the provided text
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    # Initialize the LSA summarizer
    summarizer = LsaSummarizer()
    # Get the summary with the specified number of sentences
    summary = summarizer(parser.document, sentences_count)
    # Combine the sentences into a single string
    summarized_text = " ".join([str(sentence) for sentence in summary])
    return summarized_text

def translate_to_arabic(text):
    translator = Translator()
    translated = translator.translate(text, src='en', dest='ar')
    return translated.text

def url_youtube_dow(url, key, folder='videos'):
    print("Download :: " + url)
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download(output_path=folder)
    base, ext = os.path.splitext(out_file)
    new_file = os.path.join(folder, f"{key}{ext}")

    counter = 1
    while os.path.exists(new_file):
        new_file = os.path.join(folder, f"{key}_{counter}{ext}")
        counter += 1

    os.rename(out_file, new_file)

def check_file_in_folder(folder, filename):
    file_path = os.path.join(folder, filename)
    return os.path.exists(file_path)

def convert_mp4_to_wav(mp4_file, wav_file, output_folder='wav'):
    print("Convert Mp4 to Wav")
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    # Construct the output file path
    wav_file_path = os.path.join(output_folder, wav_file)
    # Run FFmpeg command to convert MP4 to WAV
    subprocess.run(['ffmpeg', '-i', mp4_file, '-acodec', 'pcm_s16le', '-ar', '44100', wav_file_path])

def transcribe_large_audio(path):
    print("Get Text from Audio Wav")
    """Split audio into chunks and apply speech recognition"""
    # Open audio file with pydub
    sound = AudioSegment.from_wav(path)

    # Split audio where silence is 700ms or greater and get chunks
    chunks = split_on_silence(sound, min_silence_len=700, silence_thresh=sound.dBFS - 14, keep_silence=700)

    # Create folder to store audio chunks
    folder_name = "audio-chunks"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    whole_text = ""
    # Process each chunk
    for i, audio_chunk in enumerate(chunks, start=1):
        # Export chunk and save in folder
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")

        # Recognize chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # Convert to text
            try:
                # Specify language as Arabic
                text = r.recognize_google(audio_listened, language=language_arr[1])
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                print(chunk_filename, ":", text)
                whole_text += text

    # Return text for all chunks
    return whole_text

# Define your download function here
def start():
    while True:
        sleep(1)
        print("start now...")
        for i, data in enumerate(key_url_array):
            key, url, status = data
            if not status:
                key_url_array[i] = (key,url,True)
                folder = 'videos'
                filename = key+'.mp4'
                if check_file_in_folder(folder, filename):
                    pass
                else:
                    url_youtube_dow(url,key,'videos')
                    mp4_file = 'videos/' + key + '.mp4'
                    wav_file = key+'.wav'
                    output_folder = 'wav'
                    convert_mp4_to_wav(mp4_file, wav_file, output_folder)
                    result = transcribe_large_audio('wav/' + key + '.wav')
                    print(result, file=open('file/' + key + '.txt', 'w'))
                    print("Don ...")


if __name__ == '__main__':
    # Create a thread for the download function
    start_thread = threading.Thread(target=start)
    start_thread.daemon = True  # Set the thread as daemon so it will stop when the main thread stops

    # Start the download thread
    start_thread.start()

    # Start the Flask app
    app.run(debug=True)
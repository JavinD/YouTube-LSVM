from pytube import YouTube
import requests
import re

def urlify(s):

    s = re.sub(r"[^\w\s]", '', s)

    s = re.sub(r"\s+", '-', s)

    return s

username = "marquesbrownlee"
url = "https://www.youtube.com/user/" + username + "/videos"
page = requests.get(url).content
data = str(page).split(' ')
item = 'href="/watch?'
vids = [line.replace('href="', 'youtube.com') for line in data if item in line] # list of all videos listed twice


for i in range (6):
    spaces = 0
    print(vids[i]) # index the latest video
    source = YouTube(vids[i])
    en_caption = source.captions.get_by_language_code('en')

    if not en_caption:
        continue
    
    en_caption_convert_to_srt =(en_caption.generate_srt_captions())

    filename = source.title
    filename = urlify(filename)
    filename = filename + ".txt"
    

    # save the caption to a file named Output.txt
    text_file = open(filename, "w")
    text_file.write(en_caption_convert_to_srt)
    text_file.close()

    file = open(filename, "r")
    lines = file.readlines()
    file.close()

    text = ''
    for line in lines:

        if re.search('^[0-9]+$', line) is None and re.search('^[0-9]{2}:[0-9]{2}:[0-9]{2}', line) is None and re.search('^$', line) is None and re.search('^\[.*]$', line) is None:
            spaces = 0
            text += ' ' + line.rstrip('\n')

        text = text.lstrip()
        spaces = spaces + 1

        if re.search('^\[.*]$', line):
            spaces = 0

        if spaces == 4:
            text = text + ';'

    text_file = open(filename, "w")
    text_file.write(text)
    text_file.close()
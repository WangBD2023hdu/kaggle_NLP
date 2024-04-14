# Import Basis Libraries    
import pandas as pd
import os
import re
from textblob import TextBlob

# Read text
train_set = pd.read_csv(os.path.join('text', 'train.csv'))
test_set = pd.read_csv(os.path.join('text','test.csv'))

## 填充keyword和location
train_set.keyword.fillna("unknown", inplace=True)
test_set.keyword.fillna("unknown", inplace=True)

train_set.location.fillna("UNK", inplace=True)
test_set.location.fillna("UNK", inplace=True)

# 将text全部小写 使用bert其实不需要这个步骤， 有对大小写不敏感的预训练模型
train_set.text = train_set.text.str.lower()
test_set.text = test_set.text.str.lower()

# 删除HTML标签
# 导入正则表达式包


def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', text)

train_set.text = train_set.text.apply(remove_html_tags)
test_set.text = test_set.text.apply(remove_html_tags)

# 删除网页链接
def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

train_set.text = train_set.text.apply(remove_url)
test_set.text = test_set.text.apply(remove_url)

# 删除标点符号
import string

# Storing Punctuation in a Variable
punc = string.punctuation

# The code defines a function, remove_punc1, that takes a text input and removes all punctuation characters from it using
# the translate method with a translation table created by str.maketrans. This function effectively cleanses the text of punctuation symbols.
def remove_punc(text):
    return text.translate(str.maketrans('', '', punc))

train_set.text = train_set.text.apply(remove_punc)
test_set.text = test_set.text.apply(remove_punc)

# Here Come ChatWords Which i Get from a Github Repository
# 英文网络常见的缩写
# Repository Link : https://github.com/rishabhverma17/sms_slang_translator/blob/master/slang.txt
chat_words = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "B4N": "Bye For Now",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "ILU: I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A.. Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA?": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laugher",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don't care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "BFF": "Best friends forever",
    "CSL": "Can't stop laughing"
}

# 将文本先分词，然后把简写映射到全称，然后拼接成文本
def chat_conversion(text):
    new_text = []
    for i in text.split():
        if i.upper() in chat_words:
            new_text.append(chat_words[i.upper()])
        else:
            new_text.append(i)
    return " ".join(new_text)

train_set.text = train_set.text.apply(chat_conversion)
test_set.text = test_set.text.apply(chat_conversion)

def text_correct(text):
    return TextBlob(text).correct().string

train_set.text = train_set.text.apply(text_correct)
test_set.text = test_set.text.apply(text_correct)

train_set.to_csv('pre_train.csv', sep=',', index=False)
test_set.to_csv('pre_test.csv', sep=',', index=False)
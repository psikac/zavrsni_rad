import json, re
from reldi.tagger import Tagger
positive = 0
negative = 0

t=Tagger('hr')
t.authorize('bogokdu','eT%o6Ns5$q&E##ao')

data = json.loads(t.tagLemmatise(u'Jučer je bio najgori dan u mom životu.“'.encode('utf8')))['lemmas']['lemma']

dict_positive = open('positive_words_hr.txt','r', encoding='utf8')
words_positive = dict_positive.read()
dict_negative = open('negative_words_hr.txt','r', encoding='utf8')
words_negative = dict_negative.read()



for element in data:
    text = element['text']
    print(text)
    re_text = r'\b' + text + r'\b'
    if re.search(re_text, words_positive):
        positive = positive + 1
    elif re.search(re_text, words_negative):
        negative = negative + 1

print(f'Positivne riječi: {positive} Negativne riječi: {negative}')
if positive > negative:
    print("Rečenica ima pozitivan sentiment.")
elif negative > positive:
    print("Rečenica ima negativan sentiment.")
else:
    print("Rečenica ima neutralan sentiment.")
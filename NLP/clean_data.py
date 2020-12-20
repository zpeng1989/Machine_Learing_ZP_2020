import re
import sys
import unicodedata

text_data = ["   Interrobng, By Aishwaerya Henriette!!!!!!  ", "Parking And Going. By Karl Gautier.....   ", "  Today Is The nigh. By Jarek???!!!  "]

strip_whitespace = [string.strip() for string in text_data]
print(strip_whitespace)


remove_periods = [string.replace('.', '') for string in text_data]
print(remove_periods)

def capitalizer(stringstr):
    return stringstr.upper()

print([capitalizer(string) for string in text_data])


def replace_letters_with_X(stringstr):
    #print('sss')
    return re.sub(r"[a-zA-Z]", "X", stringstr)

print([replace_letters_with_X(string) for string in text_data])


punctuation = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startwith('P'))

print([string.translate(punctuation) for string in text_data])



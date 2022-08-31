
from lzma import FILTER_LZMA2


lines=['Readme', 'How to write text files in Python']
with open('checkresult.txt', 'w') as f:
    	f.writelines(lines)

lines2 = ['Readme', 'How to write text files in Python']
with open('checkresult2.txt', 'w') as f2:
    f2.writelines(lines2)


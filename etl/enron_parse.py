import re
import pandas as pd

def parse_doc(doc, allkeys):

    keys = ['Message-ID']+re.findall('\n([\w\-]+):', doc[:doc.find('\n\n')])
    keys = pd.Series(keys).drop_duplicates().tolist()

    values = []
    for a, k in enumerate(keys):
        k = k+':'
        try:
            values.append(doc[doc.find(k)+len(k):doc.find(keys[a+1])].strip())
        except:
            values.append(doc[doc.find(k)+len(k):doc.find('\n\n')].strip())
    
    d = dict(zip(keys+['Body'],values+[doc[doc.find('\n\n'):].strip()]))
    k_to_remove = set(d.keys()) - set(allkeys)
    k_to_add = set(allkeys) - set(d.keys())
    
    for k in k_to_remove:
        d.pop(k)
    for k in k_to_add:
        d[k] = ''

    keys = [k[:-1] for k in keys]
    return d

def find_allkeys(docs):
    allkeys = [re.findall('\n([\w\-]+):', doc[:doc.find('\n\n')]) for doc in docs]
    allkeys = sum(allkeys,[])
    allkeys = set(allkeys)
    allkeys.add('Message-ID')
    allkeys.add('Body')
    return allkeys

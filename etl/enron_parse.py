import re
import pandas as pd
from tqdm import tqdm

def parse_all(docs):
    allkeys = find_allkeys(docs)
    parsed = list()
    for doc in tqdm(docs):
        d = parse_doc(doc, allkeys)
        body = d['Body']
        if body.find('Forwarded by') == -1 and body.find('Original Message') == -1 and body.find('Subject:') == -1:
            body_list = map(str.strip, body.splitlines())
            d['Body'] = ' '.join(body_list) #list(filter(None, body_list))
            parsed.append(d)
    return parsed

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

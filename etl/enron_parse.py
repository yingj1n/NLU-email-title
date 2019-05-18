import re
import pandas as pd
from tqdm import tqdm
import os
import sklearn
import numpy as np

def parse_all(docs):
    allkeys = find_allkeys(docs)
    parsed = list()
    for doc in tqdm(docs):
        d = parse_doc(doc, allkeys)
        subject = d['Subject']
        body = d['Body']
        if body.find('Forwarded by') == -1 and body.find('Original Message') == -1 and body.find('Subject:') == -1 and subject.find('Re:') == -1:
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

def filter_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(axis=0)
    df = df.drop_duplicates()
    df = df[df.Subject.str.contains("Re:") == False]
    df = df[df.Subject.str.contains("RE:") == False]
    df = df[df.Subject.str.contains("re:") == False]
    df = df[df.Subject.str.contains("rE:") == False]
    df = df[df.Subject.str.contains("FW:") == False]
    df = df[df.Subject.str.contains("Fw:") == False]
    df = df[df.Subject.str.contains("fw:") == False]
    df = df[df.Subject.str.contains("fW:") == False]
    return df

def split_data(doc_path, out_dir):
    data = pd.read_csv(doc_path)
    data = data[['Body','Subject']]
    df = sklearn.utils.shuffle(data)
    train, validate, test = np.split(df.sample(frac=1), [int(.7*len(df)), int(.9*len(df))])

    out_path = os.path.dirname(out_dir)
    try:
        os.stat(out_path)
    except FileNotFoundError:
        os.mkdir(out_path)


    train.to_csv(out_path+'train.csv', header=True, index=False)
    validate.to_csv(out_path+'val.csv', header=True, index=False)
    test.to_csv(out_path+'test.csv', header=True, index=False)
    return out_path

def read_data(path):
    train = pd.read_csv(path+'train.csv')
    val = pd.read_csv(path+'val.csv')
    test = pd.read_csv(path+'test.csv')
    return train, val, test



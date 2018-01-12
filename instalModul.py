# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 15:03:38 2018
Simple Code to Install/Update necessary modules for codes in this URL:
https://taufiksutanto.blogspot.com/2018/01/easiest-social-media-analytics.html
@author: Taufik Sutanto
Notes: the code is given 'as is' & without warranty, you are responsible for your own action
"""

import subprocess

if __name__ == "__main__":
    dependencies = ['bs4','nltk','Sastrawi','scikit-learn','python-louvain','pyldavis','textblob','networkx','wordcloud']
    try:
        from tqdm import tqdm
    except:
        subprocess.call(['pip', 'install', '-U', 'tqdm', '--upgrade'])    
    print('Installing Modules (if necessary):')
    for module in tqdm(dependencies):
        try:
            subprocess.call(['pip', 'install', '-U', module, '--upgrade'])
        except:
            pass
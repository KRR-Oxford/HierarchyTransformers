# Copyright 2023 Yuan He. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTE: the 2017 Poincare paper's source code does not give full set of hypernyms
# so we made some changes to fetch all hypernyms

import re
import pandas as pd
from nltk.corpus import wordnet as wn
from tqdm import tqdm

# download wordnet if not available
try:
    wn.all_synsets
except Exception:
    import nltk
    nltk.download('wordnet')
    
def fetch_nouns():
    words = wn.words()
    nouns = set()
    for word in words:
        nouns.update(wn.synsets(word, pos='n'))
    print(len(nouns), 'nouns fetched.')
    return nouns

def fetch_hypernyms(nouns: set):
    hypernyms = []
    for noun in nouns:
        paths = noun.hypernym_paths()
        for path in paths:
            try:
                for i in range(0, len(path)-1):
                    hypernyms.append((noun, path[i]))
            except Exception:
                continue
    hypernyms = set([(x.name(), y.name()) for x, y in hypernyms])
    print(len(hypernyms), 'hyponym-hypernym pairs fetched.' )
    return hypernyms


if __name__ == '__main__':
    wn_nouns = fetch_nouns()
    wn_hypernyms = fetch_hypernyms(wn_nouns)
    wn_hypernyms = pd.DataFrame(list(wn_hypernyms), columns=['SubEntity', 'SuperEntity'])
    wn_hypernyms.to_csv("wordnet_hypernyms.tsv", sep="\t", index=False)
    # get all nouns that belong to the mammal concept
    mammal_set = set(wn_hypernyms[wn_hypernyms.SuperEntity == 'mammal.n.01'].SubEntity.unique())
    mammal_set.add('mammal.n.01')
    # select hypernym relations where both concepts are mammal
    mammal_hypernyms = wn_hypernyms[wn_hypernyms.SubEntity.isin(mammal_set) & wn_hypernyms.SuperEntity.isin(mammal_set)]
    
    with open('filtered_mammals.txt', 'r') as fin:
        filt = re.compile(f'({"|".join([l.strip() for l in fin.readlines()])})')

    filtered_mammal_hypernyms = mammal_hypernyms[~mammal_hypernyms.SubEntity.str.cat(' ' + mammal_hypernyms.SuperEntity).str.match(filt)]
    filtered_mammal_hypernyms.to_csv("wordnet_mammal_hypernyms.tsv", sep="\t", index=False)
    print(len(filtered_mammal_hypernyms), "mammal hyponym-hypernym pairs fetched.")

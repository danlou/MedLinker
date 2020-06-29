# MedLinker
ECIR 2020 - MedLinker: Medical Entity Linking with Neural Representations and Dictionary Matching

Link to paper:
https://link.springer.com/chapter/10.1007/978-3-030-45442-5_29

Note: This is a poorly documented initial release, precipitated by some requests to have access to the code. As I have more time available, and if others remain interested, I'll try to continue improving the codebase and documentation.

# Installation

After cloning this repository and moving to the root folder, follow the steps below.

### 1. Download and extract data:

This archive contains some data adapted from UMLS, please ensure you have the required [license](https://uts.nlm.nih.gov/license.html) to use it before downloading.
Download [data.zip](TBD) (153MB) from Google Drive, and then:

```
unzip data.zip
```

Check [here](https://github.com/danlou/MedLinker/blob/master/data-tree.txt) for the files you're expected to have in the data/ directory.

If data.zip is not available, the [create_umls_kb.py](https://github.com/danlou/MedLinker/blob/master/scripts/create_umls_kb.py) script should help in re-creating the UMLS data required to run MedLinker.

### 2. Download and extract models:

Download [models.zip](https://drive.google.com/file/d/1bYgO9prTKg5AQzm7xRbwFE3ZYepi7KAn/view?usp=sharing) (1.8GB) from Google Drive, and then:

```
unzip models.zip
```

Check [here](https://github.com/danlou/MedLinker/blob/master/models-tree.txt) for the files you're expected to have in the models/ directory.

### 3. Create an environment for this project:

```
conda create -n medlinker python=3.6.5 anaconda
```

### 4. Switch to this environment:

```
conda activate medlinker
```

### 5. Change the default pip version (default breaks installing dependencies):

```
pip install pip==9.0.3
```

### 6. Install dependencies:

```
pip install -r requirements.txt
```

# Usage

For this initial release, we recommend using MedLinker with the parameters defined in [medlinker.py](https://github.com/danlou/MedLinker/blob/master/medlinker.py) .

You can test if your setup is correctly configured by simply running:

```
python medlinker.py
```

After loading the models, you should see the following output:

```
{'sentence': 'Myeloid derived suppressor cells (MDSC) are immature myeloid cells with immunosuppressive activity.',
 'tokens': ['Myeloid',
  'derived',
  'suppressor',
  'cells',
  '(MDSC)',
  'are',
  'immature',
  'myeloid',
  'cells',
  'with',
  'immunosuppressive',
  'activity.'],
 'spans': [{'start': 0,
   'end': 4,
   'text': 'Myeloid derived suppressor cells',
   'st': ('T017', 1.0),
   'cui': ('C4277543', 1.0)},
  {'start': 4,
   'end': 5,
   'text': '(MDSC)',
   'st': ('T017', 0.54723495),
   'cui': ('C4277543', 0.99998283)},
  {'start': 7,
   'end': 9,
   'text': 'myeloid cells',
   'st': ('T017', 1.0),
   'cui': ('C0887899', 1.0)}]}
```

Which should be reproducible with the following code, and easily adapted for other applications:

```
from medner import MedNER
from medlinker import MedLinker
from umls import umls_kb_st21pv as umls_kb

# default models, best configuration from paper
# to experiment with different configurations, just comment/uncomment components

cx_ner_path = 'models/ContextualNER/mm_st21pv_SCIBERT_uncased/'
em_ner_path = 'models/ExactMatchNER/umls.2017AA.active.st21pv.nerfed_nlp_and_matcher.max3.p'
ngram_db_path = 'models/SimString/umls.2017AA.active.st21pv.aliases.3gram.5toks.db'
ngram_map_path = 'models/SimString/umls.2017AA.active.st21pv.aliases.5toks.map'
st_vsm_path = 'models/VSMs/mm_st21pv.sts_anns.scibert_scivocab_uncased.vecs'
cui_vsm_path = 'models/VSMs/mm_st21pv.cuis.scibert_scivocab_uncased.vecs'
cui_clf_path = 'models/Classifiers/softmax.cui.h5'
sty_clf_path = 'models/Classifiers/softmax.sty.h5'
cui_val_path = 'models/Validators/mm_st21pv.lr_clf_cui.dev.joblib'
sty_val_path = 'models/Validators/mm_st21pv.lr_clf_sty.dev.joblib'

print('Loading MedNER ...')
medner = MedNER(umls_kb)
medner.load_contextual_ner(cx_ner_path)

print('Loading MedLinker ...')
medlinker = MedLinker(medner, umls_kb)

medlinker.load_string_matcher(ngram_db_path, ngram_map_path)  # simstring approximate string matching

# medlinker.load_st_VSM(st_vsm_path)
medlinker.load_sty_clf(sty_clf_path)
# medlinker.load_st_validator(sty_val_path, validator_thresh=0.45)

# medlinker.load_cui_VSM(cui_vsm_path)
medlinker.load_cui_clf(cui_clf_path)
# medlinker.load_cui_validator(cui_val_path, validator_thresh=0.70)

s = 'Myeloid derived suppressor cells (MDSC) are immature myeloid cells with immunosuppressive activity.'
r = medlinker.predict(s)
print(r)
```

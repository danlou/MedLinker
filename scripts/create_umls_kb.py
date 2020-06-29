"""
This script expects you've followed the instructions in https://github.com/chb/py-umls to install UMLS.
"""

import json
from collections import Counter
import sqlite3

from scispacy.umls_semantic_type_tree import construct_umls_tree_from_tsv
umls_tree = construct_umls_tree_from_tsv('data/umls_semantic_type_tree.tsv')  # change to your location

umls_db_path = '/home/dan/projects/py-umls/databases/umls.db'  # change to your location
conn = sqlite3.connect(umls_db_path)
c = conn.cursor()

cui_data = {}
source_counter = Counter()
def_mismatches = set()


st21pv_sources = set(['CPT', 'FMA', 'GO', 'HGNC', 'HPO', 'ICD10',
                      'ICD10CM', 'ICD9CM', 'MDR', 'MSH', 'MTH',
                      'NCBI', 'NCI', 'NDDF', 'NDFRT', 'OMIM',
                      'RXNORM', 'SNOMEDCT_US'])


st21pv_types = set(['T005', 'T007', 'T017', 'T022', 'T031', 'T033', 'T037', 
                    'T038', 'T058', 'T062', 'T074', 'T082', 'T091', 'T092', 
                    'T097', 'T098', 'T103', 'T168', 'T170', 'T201', 'T204'])


st21pv_types_children = {}
for st in st21pv_types:
    st_node = umls_tree.get_node_from_id(st)
    st_children = set([ch.type_id for ch in umls_tree.get_children(st_node)])
    st21pv_types_children[st] = st_children


RESTRICT_ST21PV = False
NO_DEFS = False

print('Collecting info from \'descriptions\' table ...')
for row_idx, row in enumerate(c.execute('SELECT * FROM descriptions')):
    
    CUI, LAT, SAB, TTY, STR, STY = row

    source_counter[SAB] += 1


    STY = STY.split('|')
    if LAT != 'ENG':
        continue

    if RESTRICT_ST21PV:
        if SAB not in st21pv_sources:
            continue

        valid_row_sts = []
        for row_st in STY:
            if row_st in st21pv_types:
                valid_row_sts.append(row_st)
            
            else:
                for st in st21pv_types:
                    if row_st in st21pv_types_children[st]:
                        valid_row_sts.append(st)  # not row_st !
                        break

        if len(valid_row_sts) == 0:
            continue
        else:
            STY = valid_row_sts

        if len(st21pv_types.intersection(set(STY))) == 0:
            continue

    if CUI not in cui_data:
        CUI_info = {}
        CUI_info['SAB'] = SAB
        # CUI_info['TTY'] = TTY
        CUI_info['STY'] = STY

        if NO_DEFS is False:
            CUI_info['DEF'] = []
        
        CUI_info['STR'] = [STR]
        CUI_info['Name'] = '' # custom

        cui_data[CUI] = CUI_info
    
    else:
        cui_data[CUI]['STR'].append(STR)

    # source_counter[SAB] += 1

print('# CUIs:', len(cui_data))

if NO_DEFS is False:
    print('Collecting info from \'MRDEF\' table ...')
    for row_idx, row in enumerate(c.execute('SELECT * FROM MRDEF')):    
        CUI, AUI, ATUI, SATUI, SAB, DEF, SUPPRESS, CVF = row
        
        if CUI in cui_data:
            cui_data[CUI]['DEF'].append(DEF)
        else:
            def_mismatches.add(CUI)


print('Preprocessing data ...')
for cui in cui_data.keys():
    cui_data[cui]['Name'] = cui_data[cui]['STR'][0]
    cui_data[cui]['STR'] = list(set(cui_data[cui]['STR'][1:]))


print('Storing data as JSON ...')

fn = 'umls.2017AA.active'
if RESTRICT_ST21PV:
    fn += '.st21pv'
else:
    fn += '.full'

if NO_DEFS:
    fn += '.no_defs'

fn += '.json'

with open(fn, 'w') as json_f:
    json.dump(cui_data, json_f)
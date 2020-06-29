from umls import umls_kb_st21pv

for cui in umls_kb_st21pv.get_all_cuis():
    print(cui)
    input('...')

import json
from collections import Counter

class UMLS_KB(object):

    def __init__(self, umls_version):
        self.umls_data = None
        self.umls_version = umls_version

        self.load(umls_version)

    def load(self, umls_version):
        json_path = 'data/UMLS/%s.json' % umls_version
        with open(json_path, 'r') as json_f:
            self.umls_data = json.load(json_f)

    def get_sts(self, cui):
        return self.umls_data[cui]['STY']

    def get_aliases(self, cui, include_name=True):
        aliases = self.umls_data[cui]['STR']
        if include_name:
            aliases.append(self.umls_data[cui]['Name'])

        return aliases

    def get_all_cuis(self):
        # 
        return list(self.umls_data.keys())

    def get_all_stys(self):
        #   
        all_stys = set()
        for cui in self.get_all_cuis():
            for sty in self.get_sts(cui):
                all_stys.add(sty)
        return list(all_stys)

    def get_sty_sizes(self):
        # 
        sty_sizes = Counter()
        for cui in self.get_all_cuis():
            for sty in self.get_sts(cui):
                sty_sizes[sty] += 1
        return list(sty_sizes.most_common())

    def pprint(self, cui):
        cui_info = self.umls_data[cui]
        s = ''
        s += 'CUI: %s Name: %s\n' % (cui, cui_info['Name'])
        # s += 'Definition: %s\n' % '; '.join(cui_info['DEF']) 
        s += 'Aliases (%d): %s\n' % (len(cui_info['STR']), '; '.join(cui_info['STR'][:5]))
        s += 'Types: %s\n' % '; '.join(cui_info['STY'])
        print(s)

umls_kb_st21pv = UMLS_KB('umls.2017AA.active.st21pv')
umls_kb_full = UMLS_KB('umls.2017AA.active.full')


if __name__ == '__main__':
    umls_kb_st21pv.pprint('C0001097')

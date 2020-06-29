from collections import defaultdict

import numpy as np
# import faiss
import pickle

# from umls_utils import cui2st
# # from umls import umls_kb_st21pv as umls_kb


# class FaissVSM(object):

#     def __init__(self, k=10):
#         self.index = None
#         self.labels = []
#         self.k = k

#     def create(self, vecs_txt_path):
#         vectors = []
#         with open(vecs_txt_path, encoding='utf-8') as vecs_f:
#             for line_idx, line in enumerate(vecs_f):
#                 # label, vals = line.split('\t')
#                 elems = line.strip().split()
#                 label, vals = elems[0], elems[1:]
#                 self.labels.append(label)
#                 # vectors.append(np.array(list(map(float, vals.split())), dtype=np.float32))
#                 vectors.append(np.array(list(map(float, vals)), dtype=np.float32))

#                 if line_idx % 100000 == 0:
#                     print(line_idx)

#         vectors = np.vstack(vectors)
#         d = vectors.shape[1]
#         self.index = faiss.IndexFlatL2(d)
#         # self.index = faiss.IndexLSH(d, 2 * d)
#         self.index.add(vectors)

#     def load(self, index_path, labels_path):
#         self.index = faiss.read_index(index_path)
#         with open(labels_path, 'rb') as labels_f:
#             self.labels = pickle.load(labels_f)

#         # # aux
#         # self.label2cui = {l: l.split('%')[-1] for l in self.labels}

#         # self.labels_by_st = defaultdict(list)
#         # for label in self.labels:
#         #     st = cui2st(self.label2cui[label])
#         #     if st is None:  # failed to retrieve st
#         #         continue

#         #     self.labels_by_st[st].append(label)

#     # def load(self, index_path, labels_path):
#     #     self.index = faiss.read_index(index_path)
#     #     with open(labels_path, 'rb') as labels_f:
#     #         self.labels = pickle.load(labels_f)

#     #     # aux
#     #     self.label2cui = {l: l.split('%')[-1] for l in self.labels}

#     #     self.labels_by_st = defaultdict(list)
#     #     for label in self.labels:
#     #         st = cui2st(self.label2cui[label])
#     #         if st is None:  # failed to retrieve st
#     #             continue

#     #         self.labels_by_st[st].append(label)

#     def save(self, index_path, labels_path):
#         faiss.write_index(self.index, index_path)
#         with open(labels_path, 'wb') as labels_f:
#             pickle.dump(self.labels, labels_f)

#     def most_similar(self, query_vec):
#         dists, idxs = self.index.search(query_vec.reshape(1, -1), self.k)
#         dists, idxs = dists[0], idxs[0]
#         sims = [1 - d for d in dists]
#         idx_labels = [self.labels[i] for i in idxs]
        
#         r = list(zip(idx_labels, sims))
#         r = sorted(r, key=lambda x: x[1], reverse=False)
#         return r

#     # def most_similar_cuis(self, query_vec, k=10, alias_k=30, restrict_sts=[]):

#     #     sims_cuis = defaultdict(lambda:float('inf'))

#     #     q_most_similar = self.most_similar(query_vec, alias_k)

#     #     if len(restrict_sts) > 0:
#     #         relevant_labels = set()
#     #         for st in restrict_sts:
#     #             relevant_labels.update(self.labels_by_st[st])

#     #         # filter to only consider cuis belonging to given sts
#     #         q_most_similar = [(l, d) for l, d in q_most_similar if l in relevant_labels]        

#     #     for label, dist in q_most_similar:
#     #         cui = self.label2cui[label]
#     #         if dist < sims_cuis[cui]:
#     #             sims_cuis[cui] = dist

#     #     r = list(sims_cuis.items())
#     #     return sorted(r, key=lambda x: x[1], reverse=False)[:k]


class VSM(object):

    def __init__(self, vecs_path, dtype='float32', delimiter=' ', normalize=True):
        self.vecs_path = vecs_path

        if dtype == 'float32':
            self.dtype = np.float32
        elif dtype == 'float16':
            self.dtype = np.float16
        else:
            self.dtype = np.float

        self.labels = []
        self.vectors = np.array([], dtype=self.dtype)
        self.indices = {}
        self.ndims = 0

        self.load_txt(vecs_path, delimiter)

        if normalize:
            self.normalize()

    def load_txt(self, vecs_path, delimiter):
        self.vectors = []
        with open(vecs_path, encoding='utf-8') as vecs_f:
            for line_idx, line in enumerate(vecs_f):
                elems = line.strip().split(delimiter)
                self.labels.append(elems[0])
                self.vectors.append(np.array(list(map(float, elems[1:])), dtype=self.dtype))

        self.vectors = np.vstack(self.vectors)
        self.indices = {l: i for i, l in enumerate(self.labels)}
        self.ndims = self.vectors.shape[1]

    def save_txt(self, vecs_path, delimiter='\t'):
        with open(vecs_path, 'w') as vecs_f:
            for label, vec in zip(self.labels, self.vectors):
                vec_str = ' '.join([str(round(v, 6)) for v in vec.tolist()])
                vecs_f.write('%s%s%s\n' % (label, delimiter, vec_str))        

    def load_npz(self, npz_vecs_path):
        loader = np.load(npz_vecs_path)
        self.labels = loader['labels'].tolist()
        self.vectors = loader['vectors']

        self.labels_set = set(self.labels)
        self.indices = {l: i for i, l in enumerate(self.labels)}
        self.ndims = self.vectors.shape[1]

    def save_npz(self):
        npz_path = self.vecs_path.replace('.txt', '.npz')
        np.savez_compressed(npz_path,
                            labels=self.labels,
                            vectors=self.vectors)

    def normalize(self, norm='l2'):
        self.vectors = (self.vectors.T / np.linalg.norm(self.vectors, axis=1)).T

    def get_vec(self, label):
        return self.vectors[self.indices[label]]

    def similarity(self, label1, label2):
        v1 = self.get_vec(label1)
        v2 = self.get_vec(label2)
        return np.dot(v1, v2).tolist()

    def most_similar(self, vec, threshold=0.5, topn=10):
        sims = np.dot(self.vectors, vec).astype(self.dtype)
        sims_ = sims.tolist()
        r = []
        for top_i in sims.argsort().tolist()[::-1][:topn]:
            if sims_[top_i] > threshold:
                r.append((self.labels[top_i], sims_[top_i]))
        return r

    def sims(self, vec):
        return np.dot(self.vectors, np.array(vec)).tolist()


# class CUI_VSM(VSM):

#     def load_txt(self, vecs_path, delimiter='\t'):
#         self.vectors = []
#         with open(vecs_path, encoding='utf-8') as vecs_f:
#             for line_idx, line in enumerate(vecs_f):
#                 label, vals = line.strip().split('\t')  # TO-DO: fix ignored delimiter
#                 self.labels.append(label)
#                 self.vectors.append(np.array(list(map(float, vals.split())), dtype=self.dtype))

#                 if line_idx % 1000000 == 0 and line_idx >= 1000000:  # some output when loading large files
#                     print('Loading vecs - at idx %d' % line_idx)

#         self.vectors = np.vstack(self.vectors)
#         self.indices = {l: i for i, l in enumerate(self.labels)}
#         self.ndims = self.vectors.shape[1]

#         # aux
#         self.label2cui = {l: l.split('%')[-1] for l in self.labels}
        
#         self.labels_by_st = defaultdict(list)
#         for label in self.labels:
#             st = cui2st(self.label2cui[label])
#             if st is None:  # failed to retrieve st
#                 continue

#             self.labels_by_st[st].append(label)
    
#     def most_similar_cuis(self, vec, sort=True, restrict_sts=[]):

#         relevant_idxs = list(range(len(self.labels)))
#         relevant_labels = self.labels

#         if len(restrict_sts) > 0:
#             relevant_labels = []
#             for st in restrict_sts:
#                 relevant_labels += self.labels_by_st[st]
#             relevant_idxs = [self.indices[l] for l in relevant_labels]
    
#         sims = np.dot(self.vectors[relevant_idxs], vec).astype(self.dtype).tolist()

#         sims_cuis = defaultdict(float)
#         for label, score in zip(relevant_labels, sims):
#             cui = self.label2cui[label]
#             if score > sims_cuis[cui]:
#                 sims_cuis[cui] = score
        
#         r = list(sims_cuis.items())
#         if sort:
#             return sorted(r, key=lambda x: x[1], reverse=True)
#         else:
#             return r


if __name__ == '__main__':

    # p = 'models/VSMs/umls.2017AA.active.st21pv.scibert_scivocab_uncased.cuis.vecs'
    p = 'models/VSMs/umls.2017AA.active.st21pv.en_core_sci_lg.cuis.vecs'
    vsm = FaissVSM()
    # vsm.load()
    print('creating ...')
    vsm.create(p)
    print('saving ...')
    vsm.save(p.replace('.vecs', '.index'), p.replace('.vecs', '.labels'))
    # vsm = VSM(p)

import joblib
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


def load_precomputed_embeddings(precomputed_path, mm_ann, label_mapping=None):

    all_anns, all_vecs = [], []

    with open(precomputed_path, 'r') as f:
        for line in f:
            elems = line.split('\t')
            cui = elems[3]
            sty = elems[4]
            vec = np.array(list(map(float, elems[-1].split())), dtype=np.float32)

            if mm_ann == 'sty':
                all_anns.append(sty)
            elif mm_ann == 'cui':
                all_anns.append(cui)

            all_vecs.append(vec)
    
    if label_mapping is None:
        label_mapping = {a: i + 1 for i, a in enumerate(set(all_anns))}
        label_mapping['UNK'] = 0

    X = np.vstack(all_vecs)
    y = []
    for ann in all_anns:
        try:
            y.append(label_mapping[ann])
        except KeyError:
            y.append(0)

    return X, y, label_mapping

# mm_ann = 'sty' # MLP512 Acc: 0.8110184669494629 SOFTMAX Acc: 0.7777806720469078
mm_ann = 'cui'
path_precomputed_train_vecs = 'mm_st21pv.train.scibert_scivocab_uncased.precomputed'
path_precomputed_dev_vecs = 'mm_st21pv.dev.scibert_scivocab_uncased.precomputed'

print('Loading precomputed ...')
X_train, y_train, train_label_mapping = load_precomputed_embeddings(path_precomputed_train_vecs, mm_ann)
X_dev, y_dev, _ = load_precomputed_embeddings(path_precomputed_dev_vecs, mm_ann, train_label_mapping)

# clf = MLPClassifier(hidden_layer_sizes=(512,), activation='relu', solver='adam', max_iter=200, verbose=True, random_state=42)
# clf = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam', max_iter=200, verbose=True, random_state=42)
# clf = LogisticRegression(random_state=42, multi_class='multinomial', solver='sag', max_iter=200, n_jobs=4, verbose=True)

n_classes = len(set(y_train)) + 1 # UNK
# model = Sequential([
#     Dense(512, activation='relu', input_shape=(768,)),
#     Dense(n_classes, activation='softmax'),
# ])
model = Sequential([
    Dense(n_classes, activation='softmax', input_shape=(768,)),
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

es = EarlyStopping(monitor='acc', mode='max', verbose=1, min_delta=0.01, patience=10)

print('Training ...')
# clf.fit(X_train, y_train)

model.fit(
    X_train,
    to_categorical(y_train),
    epochs=100,
    batch_size=64,
    callbacks=[es],
)

print('Evaluating ...')
# y_dev_preds = mlp.predict_proba(X_dev)
# y_dev_preds = clf.predict(X_dev)

# acc = accuracy_score(y_dev, y_dev_preds)

loss, acc = model.evaluate(X_dev, to_categorical(y_dev, num_classes=len(train_label_mapping)))
print('Acc:', acc)

print('Saving model ...')
# joblib.dump(clf, 'lr_multi.%s.model.joblib' % mm_ann)
model.save('softmax.%s.model.h5' % mm_ann)
joblib.dump(train_label_mapping, 'softmax.%s.mapping.joblib' % mm_ann)

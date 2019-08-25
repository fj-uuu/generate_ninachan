
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, SimpleRNN
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import matplotlib.pyplot as plt  # 追加
import numpy as np
import random
import sys
import io


# build the model: a single LSTM
def build_model(maxlen, chars):
    print('Build model...')
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    return model


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    # 温度付きsoftmax関数
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def bind_on_epoch_end(model, text, maxlen, chars, char_indices, indices_char):
    def on_epoch_end(epoch, logs):
        # Function invoked at end of each epoch. Prints generated text.
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(text) - maxlen - 1)
        # start_index = 0
        for diversity in [0.4]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)
            # 一文字ずつ100文字まで予測
            for i in range(100):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                # 予測結果（多項分布）
                preds = model.predict(x_pred, verbose=0)[0]
                # 出力する文字の抽出
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

    return on_epoch_end


def main():
    # テキストデータの読み込み
    path = './data_imas_cg_nina2.txt'
    with io.open(path, encoding='utf-8') as f:
        text = f.read().lower()
    print('corpus length:', len(text))

    # 文字の種類
    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    # 文字の辞書を作成
    # 文字からindexを引くための辞書
    char_indices = dict((c, i) for i, c in enumerate(chars))
    # indexから文字を引くための辞書
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # maxlen=4文字単位で区切り(1シーケンス）
    maxlen = 4
    # 開始位置
    step = 1
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        # maxlenを1文として扱い、抽出
        sentences.append(text[i: i + maxlen])
        # その1文の次の文字
        next_chars.append(text[i + maxlen])
    # 学習データセット(文章の)数
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    # 3階のテンソル（文章の数, 1文, 文字の種類）
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    # 2階のテンソル（文章の数, 文字の種類）
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    model = build_model(maxlen, chars)

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # 学習経過を可視化するため
    print_callback = LambdaCallback(on_epoch_end=bind_on_epoch_end(model, text, maxlen, chars, char_indices, indices_char))

    history = model.fit(x, y,
                        batch_size=128,
                        epochs=60,
                        callbacks=[print_callback])

    # Plot Training loss & Validation Loss
    loss = history.history["loss"]
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.title("Training loss")
    plt.legend()
    plt.savefig("loss.png")
    plt.close()


main()

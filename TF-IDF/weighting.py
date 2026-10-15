
import numpy as np
from scipy.sparse import diags


TF_VARIANTS = [
    "binarna",
    "absolutna",
    "relativna",
    "logaritmicka",
    "rozsirena"
]

IDF_VARIANTS = [
    "standardna",
    "vyhladena",
    "pravdepodobnostna",
    "maximalna",
    "jednotkova"
]


def calculate_tf(matrix, tf_type):
    # Vypocet TF podla zvoleneho variantu
    tf = matrix.astype(float).tocsr()

    if tf_type == "binarna":
        tf.data[:] = 1

    elif tf_type == "absolutna":
        pass

    elif tf_type == "relativna":
        sums = np.asarray(tf.sum(axis=1)).ravel()
        sums[sums == 0] = 1
        tf = diags(1 / sums) @ tf

    elif tf_type == "logaritmicka":
        tf.data = 1 + np.log(tf.data)

    elif tf_type == "rozsirena":
        maximums = tf.max(axis=1).toarray().ravel()
        maximums[maximums == 0] = 1

        tf = diags(1 / maximums) @ tf
        tf.data = 0.5 + 0.5 * tf.data

    else:
        raise ValueError("Neznamy typ TF: " + tf_type)

    return tf.tocsr()


def calculate_idf(matrix, idf_type):
    # Vypocet IDF podla zvoleneho variantu
    N = matrix.shape[0]

    df = np.asarray((matrix > 0).sum(axis=0)).ravel()

    # Ochrana pred delenim nulou pre termy
    # ktore sa nevyskytuju v trenovacich dokumentoch
    safe_df = np.maximum(df, 1)

    if idf_type == "standardna":
        idf = np.log(N / safe_df)

    elif idf_type == "vyhladena":
        idf = np.log((N + 1) / (df + 1)) + 1

    elif idf_type == "pravdepodobnostna":
        if np.any(df == N):
            raise ValueError(
                "Pravdepodobnostna IDF nie je definovana pre df = N"
            )
        idf = np.log((N - df) / safe_df)

    elif idf_type == "maximalna":
        idf = np.log(safe_df.max() / safe_df)

    elif idf_type == "jednotkova":
        idf = np.ones(len(df))

    else:
        raise ValueError("Neznamy typ IDF: " + idf_type)

    return idf


def calculate_tfidf(matrix, tf_type, idf):
    # Kombinacia TF a IDF vah
    tf = calculate_tf(matrix, tf_type)
    return tf.multiply(idf).tocsr()

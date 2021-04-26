import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import gensim.downloader as api
#import seaborn as sns
#from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#https://developer.foursquare.com/docs/build-with-foursquare/categories/
#https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92

def create_dataframe(file, colnames):
    return pd.read_csv(file, sep='\t', header=None, names=colnames)

def calc_mean_loc(id):
    idx_df_bool = df["user_id"] == id
    idx_df = df[idx_df_bool]
    idx_lat_rad = [np.radians(i) for i in idx_df["lat"].values]
    idx_long_rad = [np.radians(i) for i in idx_df["long"].values]
    idx_z_cart = [np.cos(i) for i in idx_lat_rad]
    idx_x_cart = np.array([np.sin(i) for i in idx_lat_rad])*np.array([np.cos(i) for i in idx_long_rad])
    idx_y_cart = np.array([np.sin(i) for i in idx_lat_rad])*np.array([np.sin(i) for i in idx_long_rad])
    z_mean_cart = np.mean(idx_z_cart)
    x_mean_cart = np.mean(idx_x_cart)
    y_mean_cart = np.mean(idx_y_cart)
    lat_mean = np.rad2deg(np.arctan2(z_mean_cart, np.sqrt(y_mean_cart**2 + x_mean_cart**2)))
    long_mean = np.rad2deg(np.arctan2(y_mean_cart, x_mean_cart))
    return (lat_mean, long_mean)

def calc_freuquency_for_id(id):
    idx_df_bool = df["user_id"] == id
    idx_df = df[idx_df_bool]
    a = idx_df["venue_cat_name"].value_counts()
    return a/idx_df.shape[0]

def calc_freuqency_df():
    keys = df["venue_cat_name"].unique()
    i=0
    df_freuqency = pd.DataFrame(data=None, index=df.user_id.unique(), columns=keys)
    for user in df.user_id.unique():
        i= i+1
        print("User ", i, " of 1083")
        a = calc_freuquency_for_id(user)
        for idx, freq in enumerate(a):
            df_freuqency.at[user, a.index[idx]] = freq
    df_freuqency.fillna(0, inplace=True)
    df_freuqency.to_csv(r"Project3_Data/df_freq.csv")
    return df_freuqency

def create_listoflists():
    # TODO: Probs useless
    df1 = df[["venue_cat_name"]]
    df2 = df1.apply(lambda x: ','.join(x.astype(str)), axis=1)
    df_clean = pd.DataFrame({'clean': df2})
    listoflists = [row.split(',') for row in df_clean['clean']]
    print(listoflists[0:2])
    return listoflists

def create_vocab(arr):
    vocab_list = []
    for elem in arr:
        vocab_list.append(elem.split()[0])
    return vocab_list

def create_model(vocab):
    # TODO: CHeck what params do
    model = Word2Vec(vocab, min_count=1)
    return model

def creater_even_newer_model(vocab):
    model = api.load("glove-wiki-gigaword-50")
    model_2 = Word2Vec(size=300, min_count=1)
    model_2.build_vocab(vocab)
    total = model_2.corpus_count
    model_2.build_vocab([list(model.vocab.keys())], update=True)
    #intersect word2vec format allows for exactly what we want, but I cant use it properly yet.
    model_2.intersect_word2vec_format(model, binary=True, lockf=1.0)
    model_2.train(vocab, total_examples=total, epochs=model_2.iter)
    return model_2

def cosine_distance(model, word, target_list, num):
    # TODO: DO urself
    cosine_dict ={}
    word_list = []
    a = model[word]
    for item in target_list :
        if item != word :
            b = model[item]
            cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
            cosine_dict[item] = cos_sim
    dist_sort=sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) # in descending order
    for item in dist_sort:
        word_list.append((item[0], item[1]))
    return word_list[0:num]

def cosine_distance_calc(vec1, vec2):
    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos_sim

def cosine_distance_matrix(model, target_list):
    M_cosine = np.zeros((df.venue_cat_name.nunique(),df.venue_cat_name.nunique()))

# following only for visualization of closest word (T-SNE Vis)
def display_closestwords_tsnescatterplot(model, word, size):
    # TODO: Do yourself
    arr = np.empty((0, size), dtype='f')
    word_labels = [word]
    close_words = model.similar_by_word(word)
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()

def create_location_plot():
    #idx_df_bool = df["venue_cat_name"] == "University"
    #filtered_df = df[idx_df_bool]
    plt.scatter(x=(df['long']), y=(df['lat']), alpha=0.5, label="Data Entries")
    #plt.scatter(x=(filtered_df['long']), y=(filtered_df['lat']), color='pink', label="Bridge Entries")
    plt.scatter(x=-73.935242, y=40.730610, color='red', label="New York City")
    plt.xlabel('Longitude')
    plt.ylabel('Latidude')
    plt.legend()
    #plt.savefig('locations.png')
    plt.show()

def create_users_activity_compare_plot():
    abc = df['user_id'].value_counts()
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    f = 0
    g = 0
    for key in abc:
        if key == 100:
            a = a + 1
        if ((key > 100) and (key <= 500)):
            b = b + 1
        if key > 500 and key <= 1000:
            c = c + 1
        if key > 1000 and key <= 1500:
            d = d + 1
        if key > 1500 and key <= 2000:
            e = e + 1
        if key > 2000 and key <= 2500:
            f = f + 1
        if key > 2500:
            g = g + 1
    lister = []
    lister.append(a)
    lister.append(b)
    lister.append(c)
    lister.append(d)
    lister.append(e)
    lister.append(f)
    lister.append(g)
    print(lister)
    plt.hist(abc, bins=7)
    plt.xlabel('Number of data entries with same userID')
    plt.ylabel('Number of userIDs')
    #plt.savefig('compare_user_activity.png')
    plt.show()

def similar_users(userID, freq_df):
    maxValuesfreq_df = freq_df.max(axis=1)
    maxCatfreq_df = freq_df.idxmax(axis=1)
    #print(maxValuesfreq_df.idxmin())
    cat = maxCatfreq_df[userID]
    max_value = maxValuesfreq_df[userID]
    print("User: ", userID, ", cat: ", cat, ", max: ", max_value)
    freq_dict = {}
    i = 0
    similar_users = [None] * 5
    for user, value in maxCatfreq_df.items():
        if (user == userID):
            continue
        if (cat == value):
            tmp_diff = abs(max_value - maxValuesfreq_df[user])
            freq_dict[user] = tmp_diff
    sorted_freq_dict = dict(sorted(freq_dict.items(), key=lambda item: item[1]))
    for key in sorted_freq_dict:
        if (i == 5):
            break
        if (i == 0):
            similar_users[i] = key
            i = i + 1
        if (similar_users[i - 1] != key):
            similar_users[i] = key
            i = i + 1
    print(similar_users)
    for i in similar_users:
        None
        #print("User: ", i, ", cat: ", maxCatfreq_df[i], ", max: ", maxValuesfreq_df[i])
    return similar_users

if __name__ == '__main__':
    columns = ["user_id", "venue_id", "venue_cat_id", "venue_cat_name", "lat", "long", "tmz_offset", "utc_time"]
    df = create_dataframe("Project3_Data/dataset_NYC.txt", columns)
    user_col = ["user_id", "ls_cats", "mean_loc"]
    #df_users = -1
    #model = create_model(create_listoflists())
    #list_of_unique_venue_cat = list(df.venue_cat_name.unique())
    #print(cosine_distance(model, 'Subway', list_of_unique_venue_cat, 5))
    #display_closestwords_tsnescatterplot(model, 'Subway', 50)
    #freq_df = pd.read_csv("Project3_Data/df_freq.csv", index_col=0)
    #freq_df = calc_freuqency_df()
    #similar_users(382, freq_df)
    #print(freq_df.loc[470])
    #for users in maxValuesfreq_df.index:

    #for user in df.user_id:
    #    print(calc_mean_loc(user))
    #for column in df.columns:
     #   print("For column ", column, " unique values: ", df[column].nunique())
    #create_users_activity_compare_plot()
    #create_location_plot()
    arr = df['venue_cat_name'].unique()
    arr = create_vocab(arr)
    model = creater_even_newer_model(arr)
    print(model.most_similar("Coffee"))
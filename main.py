import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
import gensim.downloader as api
#import seaborn as sns
#from sklearn.datasets import fetch_mldata
#from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#https://developer.foursquare.com/docs/build-with-foursquare/categories/
#https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92

def create_dataframe(file, colnames):
    return pd.read_csv(file, sep='\t', header=None, names=colnames)

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
    df1 = create_mean_loc_df()
    test_id_array = [664, 472, 585, 315, 968]
    test_list = find_similar_loc_dist(test_id_array)
    entries = get_dataframe_entries(test_list)
    print(len(entries), entries.iloc[0].venue_cat_name)
    #radius = input("Enter Radius Value: ")
    #radius = float(radius)
    radius = 1
    #print(entry.venue_cat_name)
    #plt.scatter(x=(df['long']), y=(df['lat']), alpha=0.2, label="Data Entries")
    #plt.scatter(x=(filtered_df['long']), y=(filtered_df['lat']), color='pink', label="Bridge Entries")
    plt.scatter(x=df1.loc[df1['user_id'] == test_id_array[0], 'mean_long'].iloc[0], y=df1.loc[df1['user_id'] == test_id_array[0], 'mean_lat'].iloc[0], alpha=0.4, color='coral')
    plt.scatter(x=df1.loc[df1['user_id'] == test_id_array[1], 'mean_long'].iloc[0], y=df1.loc[df1['user_id'] == test_id_array[1], 'mean_lat'].iloc[0], alpha=0.4, color='purple')
    plt.scatter(x=df1.loc[df1['user_id'] == test_id_array[2], 'mean_long'].iloc[0], y=df1.loc[df1['user_id'] == test_id_array[2], 'mean_lat'].iloc[0], alpha=0.4, color='green')
    plt.scatter(x=df1.loc[df1['user_id'] == test_id_array[3], 'mean_long'].iloc[0], y=df1.loc[df1['user_id'] == test_id_array[3], 'mean_lat'].iloc[0], alpha=0.4, color='pink')
    plt.scatter(x=df1.loc[df1['user_id'] == test_id_array[4], 'mean_long'].iloc[0], y=df1.loc[df1['user_id'] == test_id_array[4], 'mean_lat'].iloc[0], alpha=0.4, color='orange')
    plt.scatter(x=df1.loc[df1['user_id'] == test_id_array[0], 'mean_long'].iloc[0], y=df1.loc[df1['user_id'] == test_id_array[0], 'mean_lat'].iloc[0], color='coral', label="Mean User 1")
    plt.scatter(x=df1.loc[df1['user_id'] == test_id_array[1], 'mean_long'].iloc[0], y=df1.loc[df1['user_id'] == test_id_array[1], 'mean_lat'].iloc[0], color='purple', label="Mean User 2")
    plt.scatter(x=df1.loc[df1['user_id'] == test_id_array[2], 'mean_long'].iloc[0], y=df1.loc[df1['user_id'] == test_id_array[2], 'mean_lat'].iloc[0], color='green', label="Mean User 3")
    plt.scatter(x=df1.loc[df1['user_id'] == test_id_array[3], 'mean_long'].iloc[0], y=df1.loc[df1['user_id'] == test_id_array[3], 'mean_lat'].iloc[0], color='pink', label="Mean User 4")
    plt.scatter(x=df1.loc[df1['user_id'] == test_id_array[4], 'mean_long'].iloc[0], y=df1.loc[df1['user_id'] == test_id_array[4], 'mean_lat'].iloc[0], color='orange', label="Mean User 5")
    #plt.scatter(x=entry.long, y=entry.lat, color='red', label="Recommended Location")
    plt.scatter(x=entries['long'], y=entries['lat'], color='red', label="Recommended Locations")
    #plt.scatter(x=(df1['mean_long']), y=(df1['mean_lat']), color = 'orange', alpha=0.5, label="Mean Locations")
    plt.xlabel('Longitude')
    plt.ylabel('Latidude')
    plt.legend()
    plt.savefig('locations52.png')
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

### Here starts the block for Task1

def create_vocab(arr):
    vocab_list = []
    for elem in arr:
        word = elem.split()[0].lower()
        if re.findall('[^A-Za-z]', word):
            if re.findall('[^A-Za-z]', word) == '-':
                word = elem.split('-')[0].lower()
            else:
                word = 'cafe'
            continue
        vocab_list.append(word)
    return vocab_list

def create_model():
    model = api.load("glove-wiki-gigaword-50")
    return model

def get_normalised_first_word(stringer):
    word = stringer.split()[0].lower()
    if re.findall('[^A-Za-z]', word):
        if re.findall('[^A-Za-z]', word) == '-':
            word = stringer.split('-')[0].lower()
        else:
            word = 'cafe'
    return word

def cosine_distance_calc(vec1, vec2):
    cos_dist = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos_dist

def cosine_distance_matrix(vocab, model):
    n = len(vocab)
    matrix = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(n):
            if (i == j):
                continue
            words1 = vocab[i].split()
            for word in words1:
                word.lower()
                if re.findall('[^A-Za-z]', word):
                    words1.remove(word)
            words2 = vocab[j].split()
            for word in words2:
                word.lower()
                if re.findall('[^A-Za-z]', word):
                    words2.remove(word)
            if (any(word in words2 for word in words1)):
                matrix_value = 1 + cosine_distance_calc(model[get_normalised_first_word(vocab[i])], model[get_normalised_first_word(vocab[j])])
                matrix_value = matrix_value / 2
                matrix[i][j] = matrix_value
    for i in range(n):
        for j in range(n):
            if (i == j):
                continue
            if (matrix[i][j] == 0):
                matrix[i][j] = cosine_distance_calc(model[get_normalised_first_word(vocab[i])], model[get_normalised_first_word(vocab[j])])
    return matrix

def save_cos_dist_matrix(vocab):
    my_model = create_model()
    matrix = cosine_distance_matrix(vocab, my_model)
    np.savetxt('similarity_matrix.csv', matrix)

def get_most_similar(matrix, vocab, word):
    index = vocab.index(word)
    sim = 0
    res_index = index
    for i in range(len(vocab)):
        if index == i:
            continue
        tmp = matrix[index][i]
        if tmp > sim:
            sim = tmp
            res_index = i
    print("For word " + word + ", " + vocab[res_index] + " is the most similar")
    return vocab[res_index]

def sort_venues_based_on_similarity(matrix, vocab, word):
    word_index = vocab.index(word)
    row = matrix[word_index]
    sort_dict = {}
    i = 0
    for similarity_value in row:
        sort_dict[vocab[i]] = similarity_value
        i += 1
    sorted_list = sorted(sort_dict.items(), key=lambda x: x[1], reverse=True)
    return_list = []
    for venue in sorted_list:
        return_list.append(venue[0])
    return return_list

### Here ends the block for Task1

### Here starts the block for Task2

def get_dataframe_entries(lister):
    #print(len(lister))
    df1 = df[df['venue_id'].isin(lister)]
    df1 = df1.drop_duplicates(subset=['venue_id'])
    #print(df1['venue_cat_name'])
    #return df1.iloc[0]
    return df1

def calc_frequency_for_id(id):
    idx_df = df[df["user_id"] == id]            # get all df entries for specific user_id
    a = idx_df["venue_cat_name"].value_counts() # get pandas series of unique venue_cat_names with # of appearances in
                                                # user specific dataframe
    return a/idx_df.shape[0]                    # return pandas series of unique venue_cat_names with # of appearances
                                                # in user specific dataframe divided by # of all df entries in user
                                                # specific dataframe

def calc_freqency_df():
    # this variable is only used to keep track of the progress
    i = 0
    # creating the dataframe without any data as it is being filled in the following loop
    df_freqency = pd.DataFrame(data=None, index=df["user_id"].unique(), columns=df["venue_cat_name"].unique())
    for user in df.user_id.unique():
        i = i+1
        print("User ", i, " of 1083")
        # call method to calculate frequency for every unique user in the dataframe
        a = calc_frequency_for_id(user)
        # iterate over the resulting pandas series and fill up the frequency dataframe
        for idx, freq in enumerate(a):
            df_freqency.at[user, a.index[idx]] = freq
    # hardly any users have entries for all unique venue category names, the current implementation keeps these
    # "not-visited" datapoints as NaN values. To keep this new dataframe clean the NaN values are replaced with a 0.
    # (Which also makes sense semantically, seeing how this user's frequency for this category is 0 if it was never
    # visited by this user.)
    df_freqency.fillna(0, inplace=True)
    # save this dataframe in the data directory so when it is being used in the future it doesn't need to be calculated
    # all over again.
    df_freqency.to_csv(r"Project3_Data/df_freq.csv")
    return df_freqency

def similar_users(userID, freq_df):
    maxValuesfreq_df = freq_df.max(axis=1)
    maxCatfreq_df = freq_df.idxmax(axis=1)
    cat = maxCatfreq_df[userID]
    max_value = maxValuesfreq_df[userID]
    print("User: ", userID, ", category: ", cat, ", value for category: ", max_value)
    freq_dict = {}
    i = 0
    similar_users = [None] * 10
    for user, value in maxCatfreq_df.items():
        if (user == userID):
            continue
        if (cat == value):
            freq_dict[user] = abs(max_value - maxValuesfreq_df[user])
    sorted_freq_dict = dict(sorted(freq_dict.items(), key=lambda item: item[1]))
    #print(sorted_freq_dict)
    for key in sorted_freq_dict:
        if (i == 10):
            break
        else:
            similar_users[i] = key
            i += 1
        #if (i == 0):
         #   similar_users[i] = key
         #   i = i + 1
        #if (similar_users[i - 1] != key):
          #  similar_users[i] = key
          #  i = i + 1
    print(similar_users)
    for i in similar_users:
        None
        #print("User: ", i, ", cat: ", maxCatfreq_df[i], ", max: ", maxValuesfreq_df[i])
    return similar_users

### Here ends the block for Task2

### Here starts the block for Task3

def calc_mean_loc_radial(id):
    # Get all dataframe entries for this specified user_id
    idx_df = df[df["user_id"] == id]
    # Create lists of all radians
    idx_lat_rad = [np.radians(i) for i in idx_df["lat"].values]
    idx_long_rad = [np.radians(i) for i in idx_df["long"].values]
    # Create list/numpy arrays of all cartesian coordinates
    idx_z_cart = [np.cos(i) for i in idx_lat_rad]
    idx_x_cart = np.array([np.sin(i) for i in idx_lat_rad]) * np.array([np.cos(i) for i in idx_long_rad])
    idx_y_cart = np.array([np.sin(i) for i in idx_lat_rad]) * np.array([np.sin(i) for i in idx_long_rad])
    # Create mean of all cartesian coordinates
    z_mean_cart = np.mean(idx_z_cart)
    x_mean_cart = np.mean(idx_x_cart)
    y_mean_cart = np.mean(idx_y_cart)
    # Convert mean values back to spherical coordinates so they can be worked with easier
    lat_mean = np.rad2deg(np.arctan2(z_mean_cart, np.sqrt(y_mean_cart ** 2 + x_mean_cart ** 2)))
    long_mean = np.rad2deg(np.arctan2(y_mean_cart, x_mean_cart))
    return (lat_mean, long_mean)

def calc_mean_lat(id):
    # Get all dataframe entries for this specified user_id
    idx_df = df[df["user_id"] == id]
    # Create numpy array from all available latitude values
    lat_array = idx_df['lat'].to_numpy()
    # Create mean latitude value from array
    lat_mean = np.mean(lat_array)
    return (lat_mean)

def calc_mean_long(id):
    # Get all dataframe entries for this specified user_id
    idx_df = df[df["user_id"] == id]
    # Create numpy array from all available longitude values
    long_array = idx_df['long'].to_numpy()
    # Create mean latitude value from array
    long_mean = np.mean(long_array)
    return (long_mean)

def calc_mean_loc(id):
    return (calc_mean_lat(id), calc_mean_long(id))

def create_mean_loc_df():
    # Create a copy of the main dataframe, so that dataframe doesn't get changed when we drop columns and duplicates
    df1 = df.copy()
    # Drop all duplicate user_id entries, the dataframe is then filled only with unique user_ids and their first entry
    df1 = df1.drop_duplicates(subset=['user_id'])
    # Create column mean_lat and mean_long and fill it with values from each user
    df1['mean_lat'] = df1.apply(lambda row: calc_mean_lat(row['user_id']), axis=1)
    df1['mean_long'] = df1.apply(lambda row: calc_mean_long(row['user_id']), axis=1)
    # Only focus on the necessary columns
    df1 = df1[['user_id','mean_lat', 'mean_long']]
    # Save dataframe so it doesn't have to be calculated all over again
    df1.to_csv(r'Project3_Data/user_means.txt', index=False)
    return df1

def get_distance(lat1, lat2, lon1, lon2):
    # Radius of the earth in kilometers
    R = 6373.0
    # Convert each coordinate to radian
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    # Calculate difference between radians
    lon_diff = lon2 - lon1
    lat_diff = lat2 - lat1
    # Calculate distance through Haversine formula
    a = np.sin(lat_diff / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(lon_diff / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def find_locations_in_radius(id_lat, id_long, radius):
    # Copy main dataframe so it doesn't get changed when we add or drop columns and duplicates
    df1 = df.copy()
    # We are dropping the duplicates of venue_id as we only need to calculate the distance once for each venue
    df1 = df1.drop_duplicates(subset=['venue_id'])
    # Adding new column that contains the distance from the venue to the specified coordinates for each venue
    df1['dist'] = df1.apply(lambda row: get_distance(row['lat'], id_lat, row['long'], id_long), axis=1)
    df1 = df1[['venue_id', 'dist']]
    # We focus only on the venues that are in our specified radius
    df1 = df1[df1['dist'] < radius]
    # Return list of all venue_ids in our radius
    loc_id_list = df1['venue_id'].to_numpy()
    return loc_id_list

def find_matching_strings(string_list):
    # Method we need to find the venue_ids (stringd) that are found in each radius (=> eligible venues)
    # Sort the list of lists string_list in descending length of inner lists
    string_list.sort(key=len, reverse=True)
    # We are using masks to iteratively eliminate venue_ids, so we need to start with the "longest" list.
    # If the longest list doesn't contain a venue_id that is in each of the inner lists then there is no venue_id
    # that can be found in each of the radii
    main_list = string_list[0]
    for i in range(1,5):
        # Check two "lists" (numpy arrays) if they have matching strings
        # Here we check every inner list with the main_list (longest list). Main_list loses elements with each iteration
        mask = np.isin(main_list, string_list[i])
        main_list = main_list[mask]
    # Return main_list (might be empty at this point)
    return main_list

def find_similar_loc_dist(id_array):
    radius = .5
    increase_rate = 0
    lister = []
    while len(lister) == 0:
        print("Radius at: ", radius+increase_rate)
        cmp_list = []
        for user in id_array:
            lat1 = calc_mean_lat(user)
            long1 = calc_mean_long(user)
            cmp_list.append(find_locations_in_radius(lat1, long1, radius + increase_rate))
        lister = find_matching_strings(cmp_list)
        increase_rate = increase_rate + 0.1
    print("Increase Rate was at ", increase_rate)
    print("Radius was at ", increase_rate+radius)
    return lister

### Here ends the block for Task3

def find_index_in_array(string, array):
    return array.index(string)

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

    #model = creater_even_newer_model(arr)
    #print(model.most_similar("Coffee"))

    #test_id_array = [470, 979, 69, 395, 87]
    #test_list = find_similar_loc_dist(test_id_array)
    #entry = get_dataframe_entries(test_list)
    #print(entry.venue_cat_name)

    #all_words = df['venue_cat_name'].unique()
    #print(all_words)
    #vocab = create_vocab(all_words)
    #vocab = list(dict.fromkeys(vocab))
    #save_cos_dist_matrix(vocab)
    #matrix = np.loadtxt('similarity_matrix.csv', usecols=range(len(vocab)))
    #word = vocab[3]
    #print(vocab)
    #get_most_similar(matrix

    #df_test = pd.DataFrame([[1100, "4cd544d894848cfa6a0de5b1", "4bf58dd8d48988d103941735", "Home (private)", 41, -74.3, -240, "Tue Apr 03 19:20:46 +0000 2012"],
    #                       [1101, "4cd544d894848cfa6a0de5b1", "4bf58dd8d48988d103941735", "Home (private)", 40.55, -74.3, -240, "Tue Apr 03 19:20:46 +0000 2012"],
    #                       [1102, "4cd544d894848cfa6a0de5b1", "4bf58dd8d48988d103941735", "Home (private)", 41, -73.7, -240, "Tue Apr 03 19:20:46 +0000 2012"],
    #                       [1103, "4cd544d894848cfa6a0de5b1", "4bf58dd8d48988d103941735", "Home (private)", 40.55, -73.7, -240, "Tue Apr 03 19:20:46 +0000 2012"],
    #                        [1104, "4cd544d894848cfa6a0de5b1", "4bf58dd8d48988d103941735", "Home (private)", 40.55, -73.7, -240, "Tue Apr 03 19:20:46 +0000 2012"]],
    #                       columns=["user_id", "venue_id", "venue_cat_id", "venue_cat_name", "lat", "long", "tmz_offset", "utc_time"])
    #df = df.append(df_test, ignore_index=True)
    #print(df[df["user_id"] == 1100])
    #my_arr = df['user_id'].unique()
    #my_arr.sort()
    #print(my_arr[-1])
    #create_location_plot()
    #print(calc_mean_loc(1103))

    vocab = df['venue_cat_name'].unique()
    #save_cos_dist_matrix(vocab)
    matrix = np.loadtxt('similarity_matrix.csv')
    vocab = vocab.tolist()
    test_word = vocab[53]
    get_most_similar(matrix, vocab, test_word)
    print(sort_venues_based_on_similarity(matrix, vocab, test_word))
    # next steps:
    # 1. implement first task: Find venues that are of ;MOST-SIMILAR; category, sort them by distance from
    # mean location of user and finally, from that list, find venues user hasn't visited yet
    # 2. implement user similarity in the way I specified it in my second submission
    # 3. fix task3
    #print(df['venue_cat_name'].nunique())
    #model = create_model()
    #print(model["American Restaurant"])




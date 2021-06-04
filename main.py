import pandas as pd
import numpy as np
import re
import gensim.downloader as api
import matplotlib.pyplot as plt

def create_dataframe(file, colnames):
    return pd.read_csv(file, sep='\t', header=None, names=colnames)

def create_task1_location_plot_scaled(userID, venue_array):
    user_lat = calc_mean_lat(userID)
    user_long = calc_mean_long(userID)

    plt.scatter(x=(df['long']), y=(df['lat']), alpha=0.1, label="Data Entries", color="pink")

    for venue in venue_array:
        entry = find_entry_in_df(venue)
        plt.scatter(x=entry["long"], y=entry["lat"], color='blue')
    plt.scatter(x=user_long, y=user_lat, color="red", label="User Mean")

    plt.title("Specified user mean and recommended locations (Task 1)")
    plt.xlabel('Longitude')
    plt.ylabel('Latidude')
    plt.legend()
    #plt.savefig('locations_task1.png')
    plt.show()

def create_task1_location_plot(userID, venue_array):
    user_lat = calc_mean_lat(userID)
    user_long = calc_mean_long(userID)

    for venue in venue_array:
        entry = find_entry_in_df(venue)
        plt.scatter(x=entry["long"], y=entry["lat"], color='blue')
    plt.scatter(x=user_long, y=user_lat, color="red", label="User Mean")

    plt.title("Specified user mean and recommended locations (Task 1)")
    plt.xlabel('Longitude')
    plt.ylabel('Latidude')
    plt.legend()
    #plt.savefig('locations_task1.png')
    plt.show()

def create_task2_location_plot(userID, user_array):
    user_lat = calc_mean_lat(userID)
    user_long = calc_mean_long(userID)

    plt.scatter(x=(df['long']), y=(df['lat']), alpha=0.1, label="Data Entries", color="pink")

    for user in user_array:
        lat = calc_mean_lat(user)
        long = calc_mean_long(user)
        plt.scatter(x=long, y=lat, color='blue')
    plt.scatter(x=user_long, y=user_lat, color="red", label="User Mean")

    plt.title("Specified user mean and similar user's means (Task 2)")
    plt.xlabel('Longitude')
    plt.ylabel('Latidude')
    plt.legend()
    #plt.savefig('locations_task2.png')
    plt.show()

def create_task3_location_plot_scaled(user_array, venue_array):
    color_array = ["purple", "green", "blue", "yellow", "magenta"]

    plt.scatter(x=(df['long']), y=(df['lat']), alpha=0.1, label="Data Entries", color="pink")

    i = 0
    for user in user_array:
        lat = calc_mean_lat(user)
        long = calc_mean_long(user)
        label_string = "Mean User " + str((i+1))
        plt.scatter(x=long, y=lat, color=color_array[i], label=label_string)
        i += 1

    for venue in venue_array:
        lat = find_entry_in_df(venue)["lat"]
        long = find_entry_in_df(venue)["long"]
        plt.scatter(x=long, y=lat, color="red")

    plt.title("User means and recommended meet-up locations (Task 3) to scale")
    plt.xlabel('Longitude')
    plt.ylabel('Latidude')
    plt.legend()
    #plt.savefig('locations_task3_scaled.png')
    plt.show()

def create_task3_location_plot(user_array, venue_array):
    color_array = ["purple", "green", "blue", "yellow", "magenta"]

    i = 0
    for user in user_array:
        lat = calc_mean_lat(user)
        long = calc_mean_long(user)
        label_string = "Mean User " + str((i+1))
        plt.scatter(x=long, y=lat, color=color_array[i], label=label_string)
        i += 1

    for venue in venue_array:
        lat = find_entry_in_df(venue)["lat"]
        long = find_entry_in_df(venue)["long"]
        plt.scatter(x=long, y=lat, color="red")

    plt.title("User means and recommended meet-up locations (Task 3)")
    plt.xlabel('Longitude')
    plt.ylabel('Latidude')
    plt.legend()
    #plt.savefig('locations_task3.png')
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
    np.savetxt('Project3_Data/similarity_matrix.csv', matrix)

def get_most_similar(matrix, word):
    vocab = df['venue_cat_name'].unique().tolist()
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

def sort_venues_based_on_similarity(matrix, word):
    vocab = df['venue_cat_name'].unique().tolist()
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

def find_venues_from_category(userID, cat):
    series1 = df.loc[df['venue_cat_name'] == cat]['venue_id']
    series2 = df.loc[df['user_id'] == userID]['venue_id']
    result_list = []
    for entry1 in series1:
        for entry2 in series2:
            if entry1 != entry2:
               result_list.append(entry1)
    result_list = list(dict.fromkeys(result_list))
    lat_user = calc_mean_lat(userID)
    long_user = calc_mean_long(userID)
    sort_dict = {}
    for entry in result_list:
        df_entry = find_entry_in_df(entry)
        sort_dict[entry] = get_distance(lat_user, df_entry["lat"], long_user, df_entry["long"])
    sort_dict = dict(sorted(sort_dict.items(), key=lambda item: item[1]))
    result_list = list(sort_dict.keys())
    return result_list

def recommend_new_locations(userID, cat, matrix):
    sorted_cat_list = sort_venues_based_on_similarity(matrix, cat)
    result_list = []
    i = 0
    while len(result_list) < 5:
        result_list.extend(find_venues_from_category(userID, sorted_cat_list[i]))
        i += 1
    if len(result_list) > 5:
        result_list = result_list[:5]
    return result_list

def find_entry_in_df(venueID):
    return df[df["venue_id"] == venueID].iloc[0]

def find_all_entries_in_df(venueID):
    return df.loc[df["venue_id"] == venueID]


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
        print("User: ", i, ", cat: ", maxCatfreq_df[i], ", max: ", maxValuesfreq_df[i])
    return similar_users

def find_users_for_category(userID, freq_df, cat):
    maxValuesfreq_df = freq_df.max(axis=1)
    maxCatfreq_df = freq_df.idxmax(axis=1)
    result_dict = {}
    # little redundant
    for i, row in freq_df.iterrows():
        user = i
        if user == userID:
            continue
        individual_cat = maxCatfreq_df[user]
        if individual_cat == cat:
            result_dict[user] = maxValuesfreq_df[user]
    focus_value = freq_df.loc[userID][cat]
    # sort users based on frequency difference to userIDs freq for this category
    # iterate over dict and calc absolute difference
    for key, value in result_dict.items():
        result_dict[key] = np.abs(focus_value-value)
    result_dict = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=True))
    return result_dict


def find_similar_users_final(userID, freq_df, matrix):
    maxValuesfreq_df = freq_df.max(axis=1)
    maxCatfreq_df = freq_df.idxmax(axis=1)
    cat = maxCatfreq_df[userID]
    max_value = maxValuesfreq_df[userID]
    print("User: ", userID, ", category: ", cat, ", value for category: ", max_value)
    similar_users = []
    for key, value in find_users_for_category(userID, freq_df, cat).items():
        similar_users.append(key)
        if len(similar_users) == 10:
            break
    if len(similar_users) < 10:
        category_list = sort_venues_based_on_similarity(matrix, cat)
        for new_cat in category_list:
            for key, value in find_users_for_category(userID, freq_df, new_cat).items():
                similar_users.append(key)
                if len(similar_users) == 10:
                    break
            if len(similar_users) == 10:
                break
    return similar_users

def quick_info_user(userID, freq_df):
    maxValuesfreq_df = freq_df.max(axis=1)
    maxCatfreq_df = freq_df.idxmax(axis=1)
    cat = maxCatfreq_df[userID]
    max_value = maxValuesfreq_df[userID]
    print("User: ", userID, ", category: ", cat, ", value for category: ", max_value)
    return

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
    df1.to_csv(r'Project3_Data/user_means.csv', index=False)
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
    radius = 2 # .5
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

def find_similar_loc_dist_filtered(id_array):
    radius = 2 # .5
    increase_rate = 0
    lister = []
    result_list = []
    while len(lister) == 0:
        print("Radius at: ", radius+increase_rate)
        cmp_list = []
        for user in id_array:
            lat1 = calc_mean_lat(user)
            long1 = calc_mean_long(user)
            cmp_list.append(find_locations_in_radius(lat1, long1, radius + increase_rate))
        lister = find_matching_strings(cmp_list)
        increase_rate = increase_rate + 0.1
        for venue in lister:
            category = find_entry_in_df(venue)["venue_cat_name"]
            if "Restaurant" in category:
                result_list.append(venue)
        if result_list == 0:
            lister = []
    print("Increase Rate was at ", increase_rate)
    print("Radius was at ", increase_rate+radius)
    return result_list

### Here ends the block for Task3

def find_index_in_array(string, array):
    return array.index(string)

if __name__ == '__main__':

    ### Necessary starts

    columns = ["user_id", "venue_id", "venue_cat_id", "venue_cat_name", "lat", "long", "tmz_offset", "utc_time"]
    df = create_dataframe("Project3_Data/dataset_NYC.txt", columns)
    user_col = ["user_id", "ls_cats", "mean_loc"]
    # freq_df = calc_freuqency_df()
    freq_df = pd.read_csv("Project3_Data/df_freq.csv", index_col=0)
    # save_cos_dist_matrix(vocab)
    matrix = np.loadtxt('Project3_Data/similarity_matrix.csv')
    # mean_loc_df = create_mean_loc_df()
    mean_loc_df = pd.read_csv('Project3_Data/user_means.csv')

    ### Setting up testing-enviornment

    test_user = 420
    test_word = "Building"

    get_most_similar(matrix, test_word)
    sorted_test_list = sort_venues_based_on_similarity(matrix, test_word)
    print(sort_venues_based_on_similarity(matrix, test_word))


    ### Task 1 Working Test

    task1_result = recommend_new_locations(test_user, test_word, matrix)
    print(task1_result)

    create_task1_location_plot(test_user, task1_result)
    create_task1_location_plot_scaled(test_user,task1_result)

    #for venue in task1_result:
    #    print(find_all_entries_in_df(venue))

    ### Task 1 Working Test End


    ### Task 2 Working Test

    task2_result = find_similar_users_final(test_user, freq_df, matrix)
    print(task2_result)

    create_task2_location_plot(test_user, task2_result)

    for user in task2_result:
        quick_info_user(user, freq_df)

    ### Task 2 Working Test End


    ### Task 3 Working Test

    task3_result = find_similar_loc_dist(task2_result[:5])
    print(task3_result)
    create_task3_location_plot(task2_result[:5], task3_result)
    create_task3_location_plot_scaled(task2_result[:5], task3_result)

    for venue in task3_result:
        print(find_entry_in_df(venue)["venue_cat_name"])

    ### Task 3 Working Test End




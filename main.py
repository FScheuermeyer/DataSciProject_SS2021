import pandas as pd
import numpy as np
import re
import gensim.downloader as api
import matplotlib.pyplot as plt

def create_dataframe(file, colnames):
    return pd.read_csv(file, sep='\t', header=None, names=colnames)

def create_task1_location_plot_scaled(userID, venue_array, filename):
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
    plt.savefig('locations_task1_scaled_' + filename + '.png')
    plt.show()

def create_task1_location_plot(userID, venue_array, filename):
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
    plt.savefig('locations_task1_' + filename + '.png')
    plt.show()

def create_task2_location_plot(userID, user_array, filename):
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
    plt.savefig('locations_task2_' + filename + '.png')
    plt.show()

def create_task3_location_plot_scaled(user_array, venue_array, filename):
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
    plt.savefig('locations_task3_scaled_' + filename + '.png')
    plt.show()

def create_task3_location_plot(user_array, venue_array, filename):
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
    plt.savefig('locations_task3_' + filename + '.png')
    plt.show()

def test_create_task3_location_plot(user_array, location):
    color_array = ["purple", "green", "blue", "yellow", "magenta"]

    i = 0
    for user in user_array:
        lat = calc_mean_lat(user)
        long = calc_mean_long(user)
        label_string = "Mean User " + str((i+1))
        plt.scatter(x=long, y=lat, color=color_array[i], label=label_string)
        i += 1

    plt.scatter(x=location[1], y=location[0], color="red")

    plt.title("User means and recommended meet-up locations (Test)")
    plt.xlabel('Longitude')
    plt.ylabel('Latidude')
    plt.legend()
    plt.savefig('locations_task3.png')
    plt.show()

### Here starts the block for Task1

def find_category_name_for_id(venue_category_ID):
    return df.loc[df["venue_cat_id"] == venue_category_ID].iloc[0]["venue_cat_name"]

def create_vocab(arr):
    # This is only a testing-method that creates a vocabulary of categories with single words
    vocab_list = []
    for elem in arr:
        # It is using the first word of each phrase as the phrase's defining word. This implementation doesn't always
        # make much sense, and should be improved.
        word = elem.split()[0].lower()
        if re.findall('[^A-Za-z]', word):
            # There are two venue category names that prove difficult to deal with: "Café" and "Gluten-free Restaurant"
            # "Café"'s last character makes assigning it to a word-vector pretty difficult, so it has to be switched
            # to "cafe"
            # "Gluten-free Restaurant"'s hyphen isn't separated by spaces, so the split() method doesn't function
            # properly
            if re.findall('[^A-Za-z]', word) == '-':
                word = elem.split('-')[0].lower()
            else:
                word = 'cafe'
            continue
        vocab_list.append(word)
    return vocab_list

def get_normalised_first_word(stringer):
    # This method is similar to the create_vocab method and and creates a "normalised" single-word category
    # It is using the first word of each phrase as the phrase's defining word. This implementation doesn't always
    # make much sense, and should be improved
    word = stringer.split()[0].lower()
    if re.findall('[^A-Za-z]', word):
        # There are two venue category names that prove difficult to deal with: "Café" and "Gluten-free Restaurant"
        # "Café"'s last character makes assigning it to a word-vector pretty difficult, so it has to be switched
        # to "cafe"
        # "Gluten-free Restaurant"'s hyphen isn't separated by spaces, so the split() method doesn't function
        # properly
        if re.findall('[^A-Za-z]', word) == '-':
            word = stringer.split('-')[0].lower()
        else:
            word = 'cafe'
    return word

def create_model():
    # this method returns the already pre-trained word-vector model
    model = api.load("glove-wiki-gigaword-50")
    return model

def cosine_distance_calc(vec1, vec2):
    # this method implements the cosine distance calculation between two vectors
    cos_dist = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos_dist

def cosine_distance_matrix(vocab, model):
    # This method is the main function that calculates the similarity between each unique venue category
    n = len(vocab)
    # The matrix gets initialized with zero-values, that is the reason why venue category names have a similarity of
    # 0 for themselves
    matrix = np.zeros(shape=(n, n))
    # This nested for-loop iterates over the entire matrix
    # The first nested loop fills in the values for venue category names that are phrases (made up of multiple words)
    # and both phrases contain the same word at least once
    for i in range(n):
        for j in range(n):
            if (i == j):
                # skip calculating the similarity between the venue category name and itself
                continue
            # split and normalize the first phrase
            words1 = vocab[i].split()
            for word in words1:
                word.lower()
                if re.findall('[^A-Za-z]', word):
                    words1.remove(word)
            # split and normalize the second phrase
            words2 = vocab[j].split()
            for word in words2:
                word.lower()
                if re.findall('[^A-Za-z]', word):
                    words2.remove(word)
            # if there is at least one word that is part of both phrase-word lists calculate the similarity with a bias
            if (any(word in words2 for word in words1)):
                # add the bias-number 1 to the calculated cosine distance
                matrix_value = 1 + cosine_distance_calc(model[get_normalised_first_word(vocab[i])], model[get_normalised_first_word(vocab[j])])
                # divide this value by 2 so it can't possibly be higher than 1
                # this seems to lower the similarity by a lot, however most calculated similarity values are pretty tiny
                # so this step still makes sense semantically
                matrix_value = matrix_value / 2
                # assign the calculated value to its corresponding spot in the matrix
                matrix[i][j] = matrix_value
    # the second nested for-loop calculates the similarities between venue category names that do not contain the same
    # words
    for i in range(n):
        for j in range(n):
            if (i == j):
                # skip calculating the similarity between the venue category name and itself
                continue
            if (matrix[i][j] == 0):
                # if the value isn't calculated yet (=> the cell was not handled in the first nested loop)
                # calculate it now
                matrix[i][j] = cosine_distance_calc(model[get_normalised_first_word(vocab[i])], model[get_normalised_first_word(vocab[j])])
    return matrix

def save_cos_dist_matrix(vocab):
    # this method calculates and saves the similarity-matrix
    my_model = create_model()
    matrix = cosine_distance_matrix(vocab, my_model)
    np.savetxt('Project3_Data/similarity_matrix.csv', matrix)
    return matrix

def get_most_similar(matrix, word):
    # Seeing how the sort_venues_based_on_similarity method does exactly the same as this method, only better, this
    # method is redundant, however it gives a good quick look on whether the recommended venue for any category makes
    # sense, so I decided to leave it in
    vocab = df['venue_cat_name'].unique().tolist()
    index = vocab.index(word)
    sim = 0
    res_index = index
    # iterate over the matrix-row that contains the specified venue category name and find the column with the highest
    # similarity-value
    for i in range(len(vocab)):
        if index == i:
            continue
        tmp = matrix[index][i]
        if tmp > sim:
            sim = tmp
            res_index = i
    # output a quick-info in the console
    print("For word " + word + ", " + vocab[res_index] + " is the most similar")
    # return the column-name
    return vocab[res_index]

def sort_venues_based_on_similarity(matrix, word):
    # this method sorts all the unique venue category names based on similarity to the specified word and returns the
    # sorted list
    vocab = df['venue_cat_name'].unique().tolist()
    word_index = vocab.index(word)
    # get the row of the matrix that contains the specified words' similarity values to other venue categories in the
    # data
    row = matrix[word_index]
    # initialize dictionary and iterator
    sort_dict = {}
    i = 0
    for similarity_value in row:
        # consider the current venue category name the key and its similarity to the specified word as the key's value
        sort_dict[vocab[i]] = similarity_value
        i += 1
    # sort this dictionary based on the keys' values
    sorted_list = sorted(sort_dict.items(), key=lambda x: x[1], reverse=True)
    # in this next step the program discards the value from the sorted list, seeing how we don't need it in the output
    return_list = []
    for venue in sorted_list:
        return_list.append(venue[0])
    return return_list

def find_venues_from_category(userID, cat):
    # this method finds venues from the specified category that are close to the specified user and to which the user
    # has never been to before
    # find all venue_ids that fit the specified category
    series1 = df.loc[df['venue_cat_name'] == cat]['venue_id']
    # find all venue_ids that the user has already been to
    series2 = df.loc[df['user_id'] == userID]['venue_id']
    result_list = []
    for entry1 in series1:
        for entry2 in series2:
            if entry1 != entry2:
                # only consider venues the user has never been to
               result_list.append(entry1)
    result_list = list(dict.fromkeys(result_list))
    # get mean location of user, so we can judge the locations based on distance to the user
    lat_user = calc_mean_lat(userID)
    long_user = calc_mean_long(userID)
    # once again I use a dictionary that maps the location as the key to its distance from the user as the key's value
    sort_dict = {}
    for entry in result_list:
        df_entry = find_entry_in_df(entry)
        sort_dict[entry] = get_distance(lat_user, df_entry["lat"], long_user, df_entry["long"])
    # sort this dictionary based on the keys' values
    sort_dict = dict(sorted(sort_dict.items(), key=lambda item: item[1]))
    # we are only interested in the keys so discard the keys' values
    result_list = list(sort_dict.keys())
    return result_list

def recommend_new_locations(userID, cat, matrix):
    # this method is the method that actually gets called to solve task1
    # first: get a list of all venue category names that are sorted based on similarity to the specified category
    sorted_cat_list = sort_venues_based_on_similarity(matrix, cat)
    # initialize a result list and iterator
    result_list = []
    i = 0
    while len(result_list) < 5:
        # add locations that could be recommended to the result-list, if less than five venues are found for the most
        # similar venue category go for the next similar category and repeat this process until we have five venues to
        # recommend
        result_list.extend(find_venues_from_category(userID, sorted_cat_list[i]))
        i += 1
    # it is possible that more than five venues get recommended when find_venues_from_category is called, this last
    # condition makes sure that only five locations are recommended
    if len(result_list) > 5:
        result_list = result_list[:5]
    # return this list
    return result_list

def find_entry_in_df(venueID):
    # this and the following method are only being used so the code is easier to read, to outsource a one line-function
    # into its own method usually doesn't make sense otherwise
    return df[df["venue_id"] == venueID].iloc[0]

def find_all_entries_in_df(venueID):
    # this method is used to test whether the recommended locations actually make sense
    return df.loc[df["venue_id"] == venueID]

def quick_info_venue(userID, venueID):
    category_name = df.loc[df["venue_id"] == venueID].iloc[0]["venue_cat_name"]
    users = df.loc[df["venue_id"] == venueID]["user_id"].unique()
    our_user_in = userID in users
    print("VenueID " + venueID + " has category: " + category_name + ". Has user " + str(userID) + " visited before? " +
          str(our_user_in))

### Here ends the block for Task1

### Here starts the block for Task2

def calc_frequency_for_id(id):
    # get all df entries for specific user_id
    idx_df = df[df["user_id"] == id]
    # get pandas series of unique venue_cat_names with # of appearances in user specific dataframe
    a = idx_df["venue_cat_name"].value_counts()
    # return pandas series of unique venue_cat_names with # of appearances in user specific dataframe divided by # of
    # all df entries in user specific dataframe
    return a/idx_df.shape[0]

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

def calc_mean_users_loc(user_array):
    lat_array = np.empty(5)
    long_array = np.empty(5)
    i = 0
    for user in user_array:
        lat_array[i] = calc_mean_lat(user)
        long_array[i] = calc_mean_long(user)
        i += 1
    mean_lat = np.mean(lat_array)
    mean_long = np.mean(long_array)
    return (mean_lat, mean_long)

def find_venues_around_loc(location):
    dist_df = df[["venue_id", "lat", "long"]].copy()
    dist_df.drop_duplicates(subset="venue_id", inplace=True)
    dist_df["dist"] = get_distance(dist_df.lat, location[0], dist_df.long, location[1])
    dist_df.sort_values("dist", inplace=True)
    return dist_df["venue_id"].iloc[:10]

### Here ends the block for Task3

def find_index_in_array(string, array):
    return array.index(string)

def task1_solution(userID, categoryID, matrix, filename):
    category = find_category_name_for_id(categoryID)
    task1_result = recommend_new_locations(userID, category, matrix)
    print(task1_result)

    create_task1_location_plot(userID, task1_result, filename)
    create_task1_location_plot_scaled(userID, task1_result, filename)

    for venue in task1_result:
        quick_info_venue(userID, venue)
    return task1_result

def task2_solution(userID, freq_df, matrix, filename):
    task2_result = find_similar_users_final(userID, freq_df, matrix)
    print(task2_result)

    create_task2_location_plot(userID, task2_result, filename)

    for user in task2_result:
        quick_info_user(user, freq_df)
    return task2_result

def task3_solution(user_array, filename):
    #task3_result = find_similar_loc_dist(user_array)
    task3_result = find_venues_around_loc(calc_mean_users_loc(user_array))
    print(task3_result)
    create_task3_location_plot(user_array, task3_result, filename)
    create_task3_location_plot_scaled(user_array, task3_result, filename)

    for venue in task3_result:
        print(find_entry_in_df(venue)["venue_cat_name"])
    return task3_result

def task3_solution_old(user_array, filename):
    task3_result = find_similar_loc_dist(user_array)
    #task3_result = test_find_venues_around_loc(test_calc_mean_users_loc(user_array))
    print(task3_result)
    create_task3_location_plot(user_array, task3_result, filename)
    create_task3_location_plot_scaled(user_array, task3_result, filename)

    for venue in task3_result:
        print(find_entry_in_df(venue)["venue_cat_name"])
    return task3_result

if __name__ == '__main__':

    ### Necessary starts

    columns = ["user_id", "venue_id", "venue_cat_id", "venue_cat_name", "lat", "long", "tmz_offset", "utc_time"]
    df = create_dataframe("Project3_Data/dataset_NYC.txt", columns)
    user_col = ["user_id", "ls_cats", "mean_loc"]
    # freq_df = calc_freuqency_df()
    freq_df = pd.read_csv("Project3_Data/df_freq.csv", index_col=0)
    # vocab = df["venue_cat_name"].unique()
    # matrix = save_cos_dist_matrix(vocab)
    matrix = np.loadtxt('Project3_Data/similarity_matrix.csv')
    # mean_loc_df = create_mean_loc_df()
    mean_loc_df = pd.read_csv('Project3_Data/user_means.csv')

    ### Setting up testing-enviornment

    test_user = 110
    test_word = "4bf58dd8d48988d1fe931735"
    filename = "test1"

    ### Task 1 Working Test

    print("Task 1 Solution:")
    get_most_similar(matrix, find_category_name_for_id(test_word))
    task1_result = task1_solution(test_user, test_word, matrix, filename)

    #for venue in task1_result:
    #    print(find_all_entries_in_df(venue))

    ### Task 1 Working Test End


    ### Task 2 Working Test

    print("Task 2 Solution:")
    task2_result = task2_solution(test_user, freq_df, matrix, filename)

    ### Task 2 Working Test End


    ### Task 3 Working Test

    print("Task 3 Solution:")
    task3_result = task3_solution(task2_result[:5], filename)
    #filename = filename + "_old"
    #task3_result_old = task3_solution_old(task2_result[:5], filename)

    ### Task 3 Working Test End




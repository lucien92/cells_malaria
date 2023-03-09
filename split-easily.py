import splitfolders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio("/home/lucien/Documents/cells_malaria/fish_data", output="/home/lucien/Documents/cells_malaria/fish_data_split",
    seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False) # default values


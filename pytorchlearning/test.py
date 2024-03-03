from read_data import MyData

root_dir = r"dataset/train"
ant_label_dir = r"ants"
ant_data_set = MyData(root_dir, ant_label_dir)

bee_label_dir = r"bees"
bee_data_set = MyData(root_dir, bee_label_dir)

train_data_set = ant_data_set + bee_data_set

img, label = ant_data_set[1]

img.show()

img, label = bee_data_set[1]

img.show()


img, label = train_data_set[1]

img.show()

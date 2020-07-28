import os

# path = "./PS3/back/Annotations"
path = "./PS3/back/JPEGImages"
path_dir = os.listdir(path)
for f in path_dir:
    file_path = "{}/{}".format(path, f)
    # os.rename(file_path, "{}/gt_img_{}".format(path, f))
    os.rename(file_path, "{}/img_{}".format(path, f))
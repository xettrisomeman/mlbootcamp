import pathlib
import os
import glob
import shutil

# get current working dir
print(os.getcwd())

# let's go inside train dir
os.chdir("./train")

# get current working dir(again)
print(os.getcwd())

# get all the pics file
file = glob.glob("*.jpg")


# collect the data
def cat_images(cat):
    if "cat" in cat:
        return cat
    return "0"


def dog_images(dog):
    if "dog" in dog:
        return dog
    return "0"


# make list of the data
total_cat_images = list(
    filter(lambda x: x != "0", sorted(list(map(cat_images, file)))))

total_dog_images = list(
    filter(lambda x: x != "0", sorted(list(map(dog_images, file)))))


# print(total_cat_images)
# move to validation folder(10k to 12.49k)
folder_cat = "cats/"
folder_dog = "dogs/"


def move_to_validation(folder_cat, folder_dog):
    for number in range(10_000, 12_500):
        for f in file:
            file_cat = f"cat.{number}.jpg"
            file_dog = f"dog.{number}.jpg"
            if file_cat in f:
                shutil.move(f"{file_cat}",
                            f"../validation/{folder_cat}{file_cat}")
            elif file_dog in f:
                shutil.move(f"{file_dog}",
                            f"../validation/{folder_dog}{file_dog}")


move_to_validation(folder_cat, folder_dog)


def move_to_train(folder_cat, folder_dog):
    for number in range(0, 10_000):
        for f in file:
            file_cat = f"cat.{number}.jpg"
            file_dog = f"dog.{number}.jpg"
            if file_cat in f:
                shutil.move(f"{file_cat}", f"./{folder_cat}{file_cat}")
            elif file_dog in f:
                shutil.move(f"{file_dog}", f"./{folder_dog}{file_dog}")


move_to_train(folder_cat, folder_dog)




from PIL import Image
from os.path import join, exists, splitext
from os import makedirs, listdir


# run this script from the root directory to convert TIF images to PNGs
# assumes that the images are located in folders with class names under ./data/

def load_convert_save_images(dir_name="Images", target_dir_name="PNGs", ext=".png"):
    
    files = listdir(dir_name)
    if not exists(target_dir_name):
        makedirs(target_dir_name)
    
    for file_name in files:
        if file_name.endswith(".tif"):
            im = Image.open(join(dir_name, file_name))
            print("Reading image: ", file_name)

            im.save(join(target_dir_name, splitext(file_name)[0]+ext))
            print("Saving image: ", splitext(file_name)[0]+ext)
            
if (__name__ == '__main__'):
    load_convert_save_images(join("data", "0_Benign"))
    load_convert_save_images(join("data", "1_Cnormal"))
    load_convert_save_images(join("data", "2_InSitu"))
    load_convert_save_images(join("data", "3_Invasive"))
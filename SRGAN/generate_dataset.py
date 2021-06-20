import os
import random
import PIL
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import argparse


parser = argparse.ArgumentParser(description='Generate synthetic LR-HR data')
parser.add_argument('--upscale', default=4, type=int, help='upscale required')
parser.add_argument('--path', default= "/content/hr_image", help='Path to the dataset')
args = parser.parse_args()



ALL_INTERPOLATIONS = [
        InterpolationMode.NEAREST,
        InterpolationMode.BILINEAR,
        InterpolationMode.BICUBIC,
        InterpolationMode.BOX,
        InterpolationMode.HAMMING,
        InterpolationMode.LANCZOS
                ]

IMG_EXTENSIONS = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]


#For storing the processed images
hr_path = "HRR_dataset/"
lr_path = "LRR_dataset/"

if not os.path.isdir(hr_path):
    os.mkdir(hr_path)

if not os.path.isdir(lr_path):
    os.mkdir(lr_path)


def check_image(filename):
    return any(filename.endswith(ext) for ext in IMG_EXTENSIONS)

def get_dimensions(img, upscale):
    '''
    input : PIL Image
    output: image with HR size multiple of the Upscale Factor needed
    '''

    w,h = img.size

    W = int(w - (w%upscale))
    H = int(h - (h%upscale))

    return W,H


if __name__ == "__main__":
	args = parser.parse_args()

	all_images = sorted([os.path.join(args.path,name) for name in os.listdir(args.path) if check_image(name)])

	for names in all_images:
	    img = PIL.Image.open(names).convert("RGB")
	    W,H = get_dimensions(img, args.upscale)                                           #Dimension now is a multiple of UPSCALE_FACTOR
	    l = len(ALL_INTERPOLATIONS)
	    img_name = os.path.splitext(names.split("/")[-1])[0]                #Extracting the name

	    lr_transformation = transforms.Compose([
	                            transforms.Resize((H//(2*args.upscale), W//((2*args.upscale))), interpolation= ALL_INTERPOLATIONS[random.randint(0, (l-1))]),
	                            transforms.Resize((H//args.upscale, W//args.upscale), interpolation= ALL_INTERPOLATIONS[random.randint(0, (l-1))]),   #Torch follows (H,W) format
	                                        ])

	    hr_transformation = transforms.Compose([
	                            transforms.CenterCrop((H,W))   
	                                            ])
	    
	    hr_img = hr_transformation(img)
	    lr_img = lr_transformation(hr_img)

	    #Saving the images
	    hr_img.save(hr_path + img_name + ".png", format= 'png')
	    lr_img.save(lr_path + img_name + ".png", format= 'png')

	print("Task Completed Succesfully!")




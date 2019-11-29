import os
import re
from collections import defaultdict
from PIL import Image 



def generate_text_file(path_to_files, file_name):
	"""
	Generates a text file with pairs of rainy images from adjacent images in time
	path_to_files [input]: path to the files whose structure will be written to the text file
	file_name [input] : file name which containts all noisy pairs
	return : none - file written to provided path
	"""
	#create a dict of lists
	dict_files = defaultdict(list)
	for root, dirs, files in os.walk(path_to_files):
		# go over each file name and split them to find out pairs
		for file in files:
			#speficy more than one delimeter in this case underscore and dash
			# only work on the ones contain 5 string pieces in total
			if len(re.split('[-_.]', file)) == 5:
				(root, time, x, y, png) = re.split('[-_.]', file)
				# create a key for each file name - use underscores to be able to decode
				key = root + '_' + x  + '_'+ y
				# create a dictionary consisting of lists
				dict_files[key].append(int(time))


	# open the file to be written
	txt_file = open(file_name,"w") 
	# traverse each key value pair in default dict
	for key,value in dict_files.items():
		# sort the frames in ascending order
		value.sort()
		# len of each list
		len_value = len(value)
		# decrypt the key into root file name, x, and y
		(root, x, y) = re.split("[_]", key)
		for i in range(1,len_value):
			# if they are adjacent we have found a pair!
			if value[i]-value[i-1] ==1:
				# file name for the pair
				str2 = '/real_world' + '/' + str(root) + '/' + str(root) + '-' + str(value[i]) + '/' + str(root) +  \
					'-' + str(value[i]) + '_' + str(x) + '_' + str(y) + '.png'
				str1 = '/real_world' + '/' + str(root) + '/' + str(root) + '-' + str(value[i-1]) + '/' + str(root) +  \
					'-' + str(value[i-1]) + '_' + str(x) + '_' + str(y) + '.png'
				txt_file.write("%s \t %s \n" %(str1,str2))

	#close the file
	txt_file.close()


def crop_images(gt, syn, size, output_dir):
	"""
	This routine crops and saves given pair of images consisting of gt (desnowed) and syn (snowed) images
	gt [input]: dir. to ground-truth image (desnowed) of random size 
	syn [input]: dir. to synthesized image (snowed) of above of the same size
	size [input]: cropping size of each image
	output_dir [input]: directory to which cropped images will be written
	return none
	"""
	# open images in rgb mode
	img_gt, img_syn = Image.open(gt, mode='r'), Image.open(syn, mode='r')
	# first check if gt and syn are of the same size
	assert (img_gt.shape == img_syn.shape, 'dimensions of gt and syn images not matching!')
	# detect origin of each image (center)
	center_x, center_y = int(im_gt.shape[0]/2.0),int(img_gt.shape[1]/2.0)
	dx, dy = int(size[0]/2.0), int(size[1]/2.0)
	# create a 4-tuple for cropping: origin is center in Cartesian move in x-y with dx,dy
	# convention: left, top, right, bottom
	(center_x-dx,center_y+dy,center_x+dx,center_y-dy) = crop_box
	# crop gt and syn
	img_gt_cropped = img_gt.crop(crop_box)
	img_syn_cropped = img_syn.crop(crop_box)
	# save the images
	img_gt_cropped.save(output_dir+gt, format = img_gt.format)
	img_syn_cropped.save(output_dir+syn, format = img.syn.format)

	

def main():
    	# specify the absolute path to the 
	path_to_files = '/home/mda/CSCI2470/real_world_rain_dataset_CVPR19/real_world'
	file_name = 'real_world_noisy_pairs.txt'
	# call the generate_text_file routine to create text file for noisy images
	generate_text_file(path_to_files, file_name)



if __name__ == '__main__':
    main()




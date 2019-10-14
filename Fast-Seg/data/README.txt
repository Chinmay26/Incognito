For more realistic background settings, I create a Mashup dataset by embedding Flickr Potrait Images [http://xiaoyongshen.me/webpage_portrait/index.html] onto MIT indoor dataset [https://web.mit.edu/torralba/www/indoor.html]. I manually choose 15 categories from the MIT Indoor dataset removing useless categories like (prison cell, green house etc.). I apply transformations (resize / shrink /random left & right align) on Flickr portrait images and embed them onto the Indoor scene dataset.

Directory Structure
-------------------

preprocessed
	|
	|__create_mashup_dataset.py  (This script creates the Potriat-Indoor dataset under ./data/processed/Potrait-Indoor/)
	|__mashup_helper.py  (Helper to create image transforms)
	|__params.json  (Config file to specify the location of Flickr/MIT datasets)
      |
      |_potrait_image_dir       // Path to the Flickr potrait dataset
      |_mask_dir                // Path to Flickr potrait segmentation masks 
      |_indoor_image_dir        // Path to MIT Indoor dataset
      |_agumented_dir           // Path to newly created mashup Portrait-Indoor dataset
      |_mask_crop_file_path     // Path to Flickr portrait crop file

processed
    |
    |__ Portrait-Indoor (directory which contains the final mashup dataset)
    		|
    		|_train
    		|_train_mask
    		|_test
    		|_test_mask
    		|_val
    		|_val_mask

raw
   |
   |_Flickr (Flickr portrait dataset)
   |_Flickr_mask (Segmentation masks of the flickr dataset)
   |_Indoor (15 categories of the MIT Indoor dataset)	


dataset.py (Dataset handler which generates Image-Mask batches)


Execute script as: python -W ignore ./data/preprocessed/create_mashup_dataset.py 


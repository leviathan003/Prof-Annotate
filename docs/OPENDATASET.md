# Open a Dataset

## Steps to open a dataset

To open a dataset, the annotator must use the keybinding, ```Ctrl+o``` to start the dir browser popup, from where they select the dataset to open.

How the dataset opens next is a scenario based response from the software, listed as below:

1. The dataset has images but not in the proper YOLO format: In this case the annotator gets a alert from the Professor that the dataset is not properly arranged in correct format, asking if they would like for him to prepare it, if selected yes, the professor prompts for further changes similar to Create a New Dataset, asking about the annotation details, modalities, train-val split after which he will prepare the dataset for the annotator and open it.

2. The dataset has labels (in correct/incorrect format) only but no image: In this case the annotator gets an alert from the Professor, stating that the dir lacks images thereby refusing to open it until the situation is fixed by the annotator.

3. The dataset is empty folder: In this case if the dir is not in the proper YOLO format, Professor will display the proper dir format to the annotator, requesting the annotator to first handle that before trying to open the dataset again.

4. The dataset has the images folder in proper format: In this case the Prof will simply ask the annotator about auto annotation, modalties, and proceed to create the data.yaml and labels folder accordingly.

5. The dataset has the proper yolo format with images and labels but lacks data.yaml: The Professor opens the dataset directly, writing out the data.yaml from the understanding that he gets from the label files.

6. The dataset is complete in its proper structure has correct format in both images and labels dir and also a data.yaml file: The Professor loads up the dataset for the annotator to start working.

## Completion
After loading the dataset, the annotator is able to view the populated dir viewer, data.yaml editor and stats panel. The canvas is populated by the first image in the train folder, which if possesses its annotations in its respective label file, will be shown as an overlay on the image. 

### Developer Note for the Interested: 
The load up of the dataset is fast and memory efficient. The entire dataset is never loaded onto the RAM of the annotator's device at once but in chunks of 21 images. However the dataset when loaded up is indexed for fast access using a dict and a sliding window algorithm that maintains the currently opened image in the canvas as the center and loads the 10 images before and after it into memory, making the hold press of ```arrow keys``` fast and reponsive by moving the loading buffer accoridngly regardless of the capability of the annotator's machine. I have tried and tested this on Intel Core i3-5005U CPU with 8GB DDR3 RAM, loading upto datasets of size 9GB without any issues. The annotator is welcome to test the software to its limits, and provide their feedback by raising any issues. The stats panel takes sometime to populate since it shows the stats of the entire dataset which until completely parsed cant be shown.

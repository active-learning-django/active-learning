## Active Learning Image Labeling

This is a labeling application that provides a labeling interface and user feedback to help the user implement active learning. This currently works on any binary classification problem, and ridge regression is done on the backend to model the data. 


This project contains a significant amount of code for multiclass classification, and the base code for creating dynamically generated data object models. which will hopefully be completed in the future.


## Image Labeling Application

### Data Flow
User can follow these steps in order to use the application:
1. Create a model shell with a title and notes/description.
2. Upload a ZIP file of images that the model will featurize.
3. User can label some initial images to train the base model.
4. After a certain amount of images are labeled, the user can view the results of the regression model by navigating to "Show Result".
5. The user can then navigate to "Uncertain Cases" to label the images the model is most unsure of.
6. The user can continue labeling the uncertain cases and rerunning the model at their convenience.
7. The user can choose to upload additional images to continue this process.

### Database 

This application runs SQLite per the default Django settings.

## Dependencies
This application uses these major Libraries/frameworks and their correspoding versions:

- Django==3.0.2
- Keras==2.2.5
- scikit-learn==0.22.2.post1
- tensorflow==1.14.0
- numpy==1.18.1
- pandas==1.0.1
- django-dynamic-model==0.1.0 

The dependencies can be installed in full by installing all libraries in requirements.txt through pip:

pip install -r requirements.txt

## Previous work
This project is based on a previous group work. Details can be found at https://github.com/tylerIams/ActiveLearningApp


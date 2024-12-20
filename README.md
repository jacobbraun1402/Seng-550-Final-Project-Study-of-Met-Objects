The Metropolitan Museum of Art, known commonly as The Met, is one of the largest art museums in the world, with millions of pieces in its collection. It is an encyclopaedic museum, meaning that it is a museum that aims to provide information on and showcase art from cultures around the world. The Met hosts a database of over 490,000 of its pieces in their collection and includes information about the pieces and their artist(s), such as the artist’s name, age, gender and nationality, as well as the culture from which that art piece came from. Each object also has an attribute signifying if the object is a highlight, meaning it is a popular or important piece. The purpose of this project was to create a prototype of an ELT pipeline and logistic regression model that could be used to predict, given a set of piece features, if that piece is a highlight. I also wanted to see if there is a bias towards or against certain kinds of art, which should not be the case if the Met aims to showcase a wide variety of art.

Requirements.txt contains all of the dependencies for this project. To install them run the following command in the directory of the cloned repository:
```
pip install -r requirements.txt
```
```extract.ipynb``` contains code for extracting objects from the Met's API. This is extremely slow, so the csv file that is in the ```extractedDataset``` folder is used for the transformation and prediction scripts.

```transform.ipynb``` contains the transformations used on the extracted dataset, as well as the building of a logistic regression model in pySpark.

```predictionPipeline.py``` is here to demonstrate how the transformations and model prototyped in ```transform.ipynb``` could be used to make predictions on other samples from the database.

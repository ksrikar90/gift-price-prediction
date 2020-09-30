# gift-price-prediction
This is a personal project to build a model to predict gift prices based time taken to restock and shelf life  during Good Friday

Steps followed to build a prediction model:

1.) Once the data is read, I build a data profile using pandas profiling to perform exploratory data analysis and understand the relations between various features and their composition

2.) I found that the dataset had four timestamp fields and since a majority of them are unique, I tried to calculate the time delta in seconds between : 

	a.) instock_date and stock_update_date
	b.) instock_date and uk_date1
	c.) instock_date and uk_date2

    In order to avoid encoding the datetime fields and still use them for prediction to improve accuracy

3.) Based on the EDA profile, I chose the features that would help build a model with most accuracy

4.) I created a base model and split training data to estimate the best hyper parameters in oder to create a final model for prediciton

5.) Once I had the required information, I trained the model and use it to predict the prices based on test data


Tools Used:

1.) Anaconda Spyder
2.) Jupyter
3.) python
4.) scikit learn

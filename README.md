# YUGAN_Individual_Project_Code_Submission
The repository for Yu Gan's Master's project Using Artificial Intelligence/Machine Learning as a means of forecasting the price of financial assets 


- The GRU_model_main.py is used to load the final GRU models selected by grid search methods, print their architecture, print their RMSE, MAE, and MAPE, then show the prediction results in the form of graphs.

- The LSTM_model_main.py is used to load the final LSTM models selected by grid search methods, print their architecture, print their RMSE, MAE, and MAPE, then show the prediction results in the form of graphs.

- The TCN_model_main.py is used to load the final GRU models selected by grid search methods, print their architecture, print their RMSE, MAE, and MAPE, then show the prediction results in the form of graphs.

- The results.py is used to plot the results of the project.

- The pic directory is used to store all the pictures in this project.

- The data directory is used to store the collected dataset and the source code for data acquisition, dataset description, and google trends acquisition.

- The Feature_Engineering directory is used to concatenate the technical indicators and the google trends and store the results after concatenating.

- The environment_set_up directory is used to clarify the dependencies and libraries needed to run the source code. This project is implemented in the Pycharm Conda environment. To rebuild the environment, One can manually pip install all the packages, libraries, and dependencies listed on the environment.yaml inside this directory. One can also try the command "conda env create -f environment.yaml" to rebuild the environment.

- The model directory is used to store the final selected models and the code to train these models. Note that the training script requires two parameters to run, the first is the stock tick label, and the second is the features. In order to input these two parameters, click run -> Edit Configurations. In the Parameters box,input two parameters. Taking META as an example, input META ['Open','High','Low','Close','AdjClose','Volume',’META’]. Then the script will run. Note that this script will run a long time to traverse all possible combinations of hyperparameters to stop.
	+ For the first parameter, one can choose from AAPL, AMZN, GOOG, MSFT, and META.
	+ For the second parameter, if running to train the model without google trends. Then the parameter is ['Open','High','Low','Close','AdjClose','Volume'].
	+ If running to train the model with google trends. Then the parameter is ['Open','High','Low','Close','AdjClose','Volume',ticker]. Where ticker is one of AAPL, AMZN, GOOG, MSFT, and META. Separate two parameters using space. Do not put any spaces within either of the parameters.

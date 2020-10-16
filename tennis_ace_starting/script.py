import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv('tennis_stats.csv')

# perform exploratory analysis here:

#plots a scatter plot of one column vs. the other
#takes two columns, and their names as arguments
def scatter_plot(col1, col2, name1, name2, title):
    plt.scatter(col1, col2)
    plt.title(title)
    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.show()
    plt.clf()

# scatter_plot(df['BreakPointsOpportunities'], df['Winnings'], 'BreakPointsOpportunities', 'Winnings', "BreakPointsOpportunities vs. Winnings")
# scatter_plot(df['FirstServePointsWon'], df['Winnings'],'FirstServePointsWon', 'Winnings', "FirstServePointsWon vs. Winnings")

## perform single feature linear regressions here:

# function to split dataset for training and testing (for the machine learning algorithm) and plots the test and prediction of the algorithm
# on a scatter plot graph
def single_linear_regression(input, outcome, input_name, outcome_name):
    slr = LinearRegression()
    #splits the dataset into training and testing 
    input_train, input_test, outcome_train, outcome_test = train_test_split(input, outcome, train_size=0.8)
    #uses training sets to train machine learning algorithm
    slr.fit(input_train, outcome_train)
    #uses input_test to predict outcome
    outcome_predict = slr.predict(input_test)
    #prints score onto console (coefficient of determination R^2)
    print("Test score of predicted "+outcome_name.lower()+" with "+input_name+" as input: "+str(slr.score(input_test, outcome_test)))
    #plots test and prediction onto scatter plot graph
    scatter_plot(outcome_test, outcome_predict, "Test", "Prediction", outcome_name+" Test vs. Prediction with "+input_name+" as Input")

# function to get best coefficient of determination R^2 (closest to 1)
# returns name of input column and outcome column that produce the best
# coefficient of determination
def get_best_fit(data):
    best_fit = test_all_combinations(data)
    return best_fit['input'], best_fit['outcome']

# returns the dictionary with the highest score (R^2) from list of
# scores
def get_highest_score(scores):
    highest_score = 0
    to_return = {}
    for i in range(len(scores)):
        if scores[i]["score"]>highest_score:
            to_return = scores[i]
            highest_score = scores[i]["score"]
    return to_return


#tests all different combinations of the different categories within the
#dataset
def test_all_combinations(data):
    categories = data.columns.tolist()
    categories.remove("Player")
    categories.remove("Year")
    scores = []
    for i in range(len(categories)):
        for j in range(len(categories)):
            if categories[i]==categories[j]:
                scores.append({"input": categories[i], "outcome": categories[j], "score": 0})
                continue
            else:
                slr = LinearRegression()
                in_data = data[[categories[i]]]
                out_data = data[[categories[j]]]
                input_train, input_test, outcome_train, outcome_test = train_test_split(in_data, out_data, train_size=0.8)
                slr.fit(input_train, outcome_train)
                scores.append({"input": categories[i], "outcome": categories[j], "score": slr.score(input_test, outcome_test)})
    
    return get_highest_score(scores)

input, outcome = get_best_fit(df)
single_linear_regression(df[[input]], df[[outcome]], input, outcome)

## perform two feature linear regressions here:
def two_feature_lin_reg(input, outcome, input_name, output_name):
    single_linear_regression(input, outcome, input_name, output_name)

two_feature_lin_reg(df[['BreakPointsOpportunities', 'FirstServeReturnPointsWon']], df[['Winnings']], 'BreakPointsOpportunities and FirstServeReturnPointsWon', 'Winnings')

## perform multiple feature linear regressions here:
def multiple_features_lin_reg(input, outcome, input_name, output_name):
    single_linear_regression(input, outcome, input_name, output_name)

multiple_inputs = df[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
'TotalServicePointsWon']]
multiple_features_lin_reg(multiple_inputs, df[["Winnings"]], "Multiple features", "Winnings")
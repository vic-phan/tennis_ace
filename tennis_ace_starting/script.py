import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv('tennis_stats.csv')
print(df.columns)


# perform exploratory analysis here:

# BreakPointsOpportunities and Winnings
plt.scatter(df['BreakPointsOpportunities'], df['Winnings'])
plt.title("BreakPointsOpportunities vs. Winnings")
plt.xlabel("BreakPointsOpportunities")
plt.ylabel("Winnings")
plt.show()
plt.clf()

plt.scatter(df['FirstServePointsWon'], df['Winnings'])
plt.title("FirstServePointsWon vs. Winnings")
plt.xlabel("FirstServePointsWon")
plt.ylabel("Winnings")
plt.show()
plt.clf()


## perform single feature linear regressions here:
# Single Feature Linear Regression on
# BreakPointsOpportunities and Winnings
# BPO = input, Winnings = outcome
slr = LinearRegression()
BPO = df[['BreakPointsOpportunities']]
Win = df[['Winnings']]
BPO_train, BPO_test, Win_train, Win_test = train_test_split(BPO, Win, train_size=0.8)
slr.fit(BPO_train, Win_train)
Win_predict = slr.predict(BPO_test)
print("Test score of predicted winnings with BreakPointsOpportunities as input: "+str(slr.score(BPO_test, Win_test)))

#Plot data vs. prediction
plt.scatter(Win_test, Win_predict, alpha=0.4)
plt.title("Winnings test vs. prediction with BreakPointOpportunities as input")
plt.xlabel("Test")
plt.ylabel("Prediction")
plt.show()
plt.clf()

# Single Feature Linear Regression on
# FirstServePointsWon and Winnings
# FSP = input, Winnings = outcome
slr2 = LinearRegression()
FSP = df[['FirstServePointsWon']]
FSP_train, FSP_test, Win_train, Win_test = train_test_split(FSP, Win, train_size=0.8)
slr2.fit(FSP_train, Win_train)
Win_predict = slr.predict(FSP_test)
print("Test score of predicted winnings with FirstServePointsWon as input: "+str(slr.score(FSP_test, Win_test)))

#Plot test vs. prediction
plt.scatter(Win_test, Win_predict, alpha=0.4)
plt.title("Winnings test vs. prediction with FirstServePointsWon as input")
plt.xlabel("Test")
plt.ylabel("Prediction")
plt.show()
plt.clf()

## perform two feature linear regressions here:






















## perform multiple feature linear regressions here:

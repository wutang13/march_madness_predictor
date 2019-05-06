# march_madness_predictor
Using machine learning to predict college basketball outcomes.

Project consists of two scripts:

get_team_data is a webscraping script that points to sports-reference.com to get data
for each team that has played in the NCAA College Basketball tournament in a given year.
It creates a csv file and writes data it finds for both the winning and losing team of a game
to one row. The targets are a one or a zero indicating either Team A won (0) or Team B won (1).
The winning team is written first on an alternating basis in order maintain a relatively even number
of datapoints for both targets. The raw data of everything that is scraped is located in ncaa.csv. 
The formatted data with all datapoints containing null entries removed are contained in the other .csv files.
Targets with scores for each team were also recorded in seperate files.

score_predictor contains two function: one for predicting the score of team A and one for predicting the winner
between two teams. Both use a Support Vector Machine for either a binary classification or regression problem.
The binary classifier is significantly more accurate the the score predictor predicting the correct team ~83% of
the time when verified with cross validation.

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from tensorflow import keras
import tensorflow as tf
from get_team_data import get_bracket


class Game:
    def __init__(self, team_1, team_2):
        self.team_1 = team_1
        self.team_2 = team_2
        self.winner = -1


def predict_score_regressor():
    cbb_data = np.genfromtxt("Data/ncaa_data_score.csv", delimiter=',')
    cbb_targets = np.genfromtxt("Data/ncaa_targets_score.csv", delimiter=",")

    scaler = StandardScaler()
    scaler.fit(cbb_data)
    scaled_cbb_data = scaler.transform(cbb_data)

    x_train, x_test, y_train, y_test = train_test_split(scaled_cbb_data, cbb_targets)

    svm = SVR(kernel='rbf', C=100, gamma=0.001).fit(x_train, y_train)
    svm.fit(x_train, y_train)

    print("Training Accuracy: {}".format(svm.score(x_train, y_train)))
    print("Testing Accuracy: {}".format(svm.score(x_test, y_test)))

    pred = svm.predict(x_test)

    game_count = 0
    correct = 0
    incorrect = 0

    while game_count < len(pred):
        if pred[game_count] > pred[game_count+1] and y_test[game_count] > y_test[game_count+1]:
            correct += 1
        elif pred[game_count] < pred[game_count + 1] and y_test[game_count] < y_test[game_count+1]:
            correct += 1
        else:
            incorrect += 1

        game_count += 2

    print("Correct:{} Incorrect:{} Precentage Correct: {}".format(correct, incorrect, (correct/(correct+incorrect))))


def predict_winner_classifier():
    cbb_data = np.genfromtxt("Data/ncaa_data_classifier.csv", delimiter=',')
    cbb_targets = np.genfromtxt("Data/ncaa_targets_classifier.csv", delimiter=",")

    scaler = StandardScaler()
    scaler.fit(cbb_data)
    scaled_cbb_data = scaler.transform(cbb_data)

    # x_train, x_test, y_train, y_test = train_test_split(scaled_cbb_data, cbb_targets)

    svm = SVC(kernel='rbf', C=100, gamma=0.001)

    print("SVM Cross Validation Accuracy: {}".format(cross_val_score(svm, scaled_cbb_data, cbb_targets, cv=10).mean()))


def predict_winner_tensorflow():
    cbb_data = np.genfromtxt("Data/ncaa_data_classifier.csv", delimiter=',')
    cbb_targets = np.genfromtxt("Data/ncaa_targets_classifier.csv", delimiter=",")

    scaler = StandardScaler()
    scaler.fit(cbb_data)
    scaled_cbb_data = scaler.transform(cbb_data)

    model = keras.Sequential([
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)]
    )

    model.compile(
        optimizer='adam',
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )

    x_train, x_test, y_train, y_test = train_test_split(scaled_cbb_data, cbb_targets)

    model.fit(x_train, y_train, epochs=10)
    test_loss, test_acc = model.evaluate(x_test, y_test)

    print("Accuracy: {}".format(test_acc))


def create_bracket_round(teams):
    game_index = 0
    team_count = len(teams)

    games = []

    while game_index < team_count:
        games.append(Game(teams[game_index], teams[game_index+1]))
        game_index += 2

    return games


def create_round_dataset(games):
    first_game = games[0].team_1.stats + games[0].team_2.stats
    game_array = [first_game]
    dataset = np.array(game_array)

    for game in games[1:]:
        dataset = np.append(dataset, [game.team_1.stats + game.team_2.stats], axis=0)

    return dataset.astype(np.float)


def predict_bracket(training_data_file, targets_file):
    cbb_data = np.genfromtxt(training_data_file, delimiter=',')
    cbb_targets = np.genfromtxt(targets_file, delimiter=",")

    # Remove year that is being predicted from training data
    cbb_data = cbb_data[64:]
    cbb_targets = cbb_targets[64:]

    scaler = StandardScaler()
    scaler.fit(cbb_data)
    scaled_training_data = scaler.transform(cbb_data)

    svm = SVC(kernel='rbf', C=100, gamma=0.001)
    svm.fit(scaled_training_data, cbb_targets)

    print("Getting Bracket Data...")
    teams = get_bracket(2019)
    print("Bracket Data Retrieved!")

    teams_in = teams

    for i in range(6):
        games_in_round = create_bracket_round(teams_in)
        round_dataset = create_round_dataset(games_in_round)

        scaled_tournament_data = scaler.transform(round_dataset)

        round_predictions = svm.predict(scaled_tournament_data)
        round_winners = {}

        for pred, game in zip(round_predictions, games_in_round):
            if pred:
                round_winners[game.team_2.name] = game.team_2
            else:
                round_winners[game.team_1.name] = game.team_1

            game.winner = pred

        print(",".join(list(round_winners.keys())))
        teams_in = list(round_winners.values())


if __name__ == '__main__':
    predict_bracket("Data/ncaa_data_classifier.csv", "Data/ncaa_targets_classifier.csv")

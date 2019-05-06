from numpy import genfromtxt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC


def predict_score_regressor():
    cbb_data = genfromtxt("ncaa_data_score.csv", delimiter=',')
    cbb_targets = genfromtxt("ncaa_targets_score.csv", delimiter=",")

    scaler = StandardScaler()
    scaler.fit(cbb_data)
    scaled_cbb_data = scaler.transform(cbb_data)

    x_train, x_test, y_train, y_test = train_test_split(scaled_cbb_data, cbb_targets)

    # c_params = [0.001, 0.01, 0.1, 1.0, 10, 100]
    # gamma_params = [0.001, 0.01, 0.1, 1.0, 10, 100]

    # param_grid = {"C": c_params, "gamma": gamma_params}

    # grid_search = GridSearchCV(SVR(), param_grid, cv=5)
    # grid_search.fit(x_train, y_train)

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


def predict_score_classifier():
    cbb_data = genfromtxt("ncaa_data_classifier.csv", delimiter=',')
    cbb_targets = genfromtxt("ncaa_targets_classifier.csv", delimiter=",")

    scaler = StandardScaler()
    scaler.fit(cbb_data)
    scaled_cbb_data = scaler.transform(cbb_data)

    '''c_params = [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
    gamma_params = [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000]

    param_grid = {"C": c_params, "gamma": gamma_params}

    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    grid_search.fit(x_train, y_train)'''

    # x_train, x_test, y_train, y_test = train_test_split(scaled_cbb_data, cbb_targets)

    svm = SVC(kernel='rbf', C=100, gamma=0.001)

    print("Cross Validation Accuracy: {}".format(cross_val_score(svm, scaled_cbb_data, cbb_targets, cv=10).mean()))


if __name__ == '__main__':
    predict_score_classifier()
import re

from bs4 import BeautifulSoup
import requests
import lxml


class Team:
    """Holds data for a particular team when predicting a bracket"""
    def __init__(self, name, stats):
        self.name = name
        self.stats = stats


# Extract the stats for a team and write it to a csv file
# Current format works for years 2017 and older
def get_team_stats(url):
    team_page = requests.get(url)
    team_soup = BeautifulSoup(team_page.content, "lxml")

    _team_stats_index = 1
    stats_table = team_soup.find_all("tbody")[_team_stats_index]
    stats_row = stats_table.find_next()

    formatted_stats = ""

    summary = team_soup.select("div[data-template]")[0].find_all_next("p")

    # SOS = Strength of Schedule
    # Standard indexing for most of the pages
    _conference_tag_index = 1
    _sos_tag_index = 6
    _sos_content_index = 1

    # Check for irregular page organization
    if summary[_conference_tag_index].contents[0].string != "Conference:":
        _conference_tag_index = 0
        _sos_tag_index = 5

    conference = summary[_conference_tag_index].find_next('a').string
    sos = summary[_sos_tag_index].contents[1]

    sos_extra_start = sos.find('(')
    sos = sos[:sos_extra_start]

    formatted_stats += conference + "," + sos.strip()

    # print(formatted_stats)

    for child in stats_row.contents:
        if child.string is not None and child.string != "Team":
            formatted_stats = formatted_stats + child.string + ","
        elif child.string != "Team":
            formatted_stats = formatted_stats + " ,"

    return formatted_stats


def get_training_data(year):
    # get the content of a ncaa bracket
    url = "https://www.sports-reference.com/cbb/postseason/{}-ncaa.html".format(year)
    bracket_page = requests.get(url)
    bracket_soup = BeautifulSoup(bracket_page.content, "lxml")

    # Initialize list of teams into winners and losers of each game
    winners = bracket_soup.find_all(class_="winner")
    losers = []

    # Find all the losing teams
    for winner in winners:
        if winner.find_next_siblings()[0].name == "div":
            losers.append(winner.find_next_siblings()[0])
        else:
            losers.append(winner.find_previous_sibling("div"))

    f = open("Data/ncaa_data.csv", "a+")

    # winner_scores = []
    # loser_scores = []

    game_count = 1
    winner_first = 0

    # Print stats to file
    for winner, loser in zip(winners, losers):
        # winner_score = winner.find_next("a").find_next().string
        # winner_scores.append(winner_score)

        winner_data = get_team_stats("https://www.sports-reference.com{}".format(winner.find_next("a")["href"]))
        loser_data = get_team_stats("https://www.sports-reference.com{}".format(loser.find_next("a")["href"]))

        # loser_score = loser.find_next("a").find_next().string
        # loser_scores.append(loser_score)

        if winner_first:
            f.write(loser_data + winner_data + "1" + "\n")
            winner_first = 0
        else:
            f.write(winner_data + loser_data + "0" + "\n")
            winner_first = 1

        print("{} Games Written to File ".format(game_count))
        game_count += 1

    f.close()


# Used to get the initial information in order to predict a bracket
def get_bracket(year):
    url = "https://www.sports-reference.com/cbb/postseason/{}-ncaa.html".format(year)
    bracket_page = requests.get(url)
    bracket_soup = BeautifulSoup(bracket_page.content, "lxml")

    bracket_rounds = bracket_soup.find_all(class_="round")

    _first_round_indices = [0, 5, 10, 15]
    first_round = []

    for index in _first_round_indices:
        first_round.append(bracket_rounds[index])

    schools = []
    for division in first_round:
        schools += division.find_all(href=re.compile("schools"))

    teams = []
    for school in schools:
        team_data = get_team_stats("https://www.sports-reference.com{}".format(school["href"]))
        split_data = team_data.split(",")

        # Remove Conference Feature
        filtered_data = split_data[1:]
        # Remove Minutes Played Feature
        filtered_data.pop(1)
        # Remove Offensive/Defensive Rebounds Feature
        filtered_data.pop(13)
        filtered_data.pop(13)
        # Remove Personal Fouls Feature
        filtered_data.pop(18)
        # Remove blank item at end
        filtered_data.pop()

        teams.append(Team(school.string, filtered_data))

    for team in teams:
        print(team.name, team.stats)

    return teams


if __name__ == '__main__':
    get_bracket(2019)

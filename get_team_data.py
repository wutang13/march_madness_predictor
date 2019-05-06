from bs4 import BeautifulSoup
import requests
import lxml


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


def get_tournament_stats(year):
    # get the content of a ncaa bracket
    url = "https://www.sports-reference.com/cbb/postseason/{}-ncaa.html".format(year)
    bracket_page = requests.get(url)
    bracket_soup = BeautifulSoup(bracket_page.content, "html.parser")

    # Initialize list of teams into winners and losers of each game
    winners = bracket_soup.find_all(class_="winner")
    losers = []

    # Find all the losing teams
    for winner in winners:
        if winner.find_next_siblings()[0].name == "div":
            losers.append(winner.find_next_siblings()[0])
        else:
            losers.append(winner.find_previous_sibling("div"))

    f = open("ncaa_data.csv", "a+")

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


if __name__ == '__main__':
    stats_year = 2019

    while stats_year > 1989:
        get_tournament_stats(stats_year)
        print("{} Data Written".format(stats_year))
        stats_year -= 1

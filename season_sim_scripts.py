# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:32:13 2021

@author: whetz
"""

import pandas as pd
import numpy as np
from numpy import random
import seaborn as sns
import itertools


def get_data(filepath):
    data = pd.read_csv(filepath)
    return data


def get_teams(data):
    teams = list(data.team1)
    teams.extend(data.team2)
    teams = list(set(teams))
    return teams


def get_team_points(data):
    temp_pts = data.dropna()
    max_week = temp_pts.week.max()
    v = temp_pts.week.max()
    temp1 = temp_pts[["week", "team1", "pts1"]].rename(
        {"team1": "team", "pts1": "pts"}, axis=1
    )
    temp2 = temp_pts[["week", "team2", "pts2"]].rename(
        {"team2": "team", "pts2": "pts"}, axis=1
    )
    temp = pd.concat([temp1, temp2])
    df = pd.pivot(data=temp, index="week", columns="team", values="pts").reset_index()
    df = df.drop(columns=["week"])
    return df


def get_records(data, team_df):
    temp_pts = data.dropna()
    max_week = temp_pts.week.max()
    temp_pts["winner"] = [
        temp_pts.team1.loc[i]
        if temp_pts.pts1.loc[i] > temp_pts.pts2.loc[i]
        else temp_pts.team2.loc[i]
        for i in range(len(temp_pts))
    ]
    records = (
        temp_pts.winner.value_counts()
        .reset_index()
        .rename({"index": "Team", "winner": "W"}, axis=1)
    )
    records["L"] = max_week - records.W
    pts = []
    for team in records.Team:
        pts.append(team_df[team].sum())
    records["pts"] = pts
    records = records.sort_values(["W", "pts"], ascending=False).reset_index(drop=True)
    return records


def get_schedule(data):
    schedule = (
        data[data.pts1.isna()].drop(columns=["pts1", "pts2"]).reset_index(drop=True)
    )
    return schedule


def get_likelihood_dict(teams, team_df):

    team_dict = {}
    for team in teams:
        mean = round(team_df[team].mean(), 3)
        std = round(team_df[team].std(), 3)
        team_dict[team] = [mean, std]
    return team_dict


def get_points(team1, team2, df):
    # Set Priors
    mu0 = round(np.array(df).mean(), 3)
    sig0 = round(np.array(df).std(), 3)

    # Get likelihood parameters for team1
    mu = round(df[team1].mean(), 3)
    sig = round(df[team1].std(), 3)

    # Update Priors and get posterior predictive parameters for team1
    muN1 = round(
        (mu0 * sig ** 2 + len(df) * mu * sig0 ** 2) / (sig ** 2 + len(df) * sig0 ** 2),
        3,
    )
    sigN1 = round(
        (((sig0 ** 2) * (sig ** 2)) / (sig ** 2 + len(df) * sig0 ** 2)) ** 0.5, 3
    )
    sigNpred1 = (sig ** 2 + sigN1 ** 2) ** 0.5

    # Get likelihood parameters for team2
    mu = round(df[team2].mean(), 3)
    sig = round(df[team2].std(), 3)

    # Update Priors and get posterior predictive parameters for team1
    muN2 = round(
        (mu0 * sig ** 2 + len(df) * mu * sig0 ** 2) / (sig ** 2 + len(df) * sig0 ** 2),
        3,
    )
    sigN2 = round(
        (((sig0 ** 2) * (sig ** 2)) / (sig ** 2 + len(df) * sig0 ** 2)) ** 0.5, 3
    )
    sigNpred2 = (sig ** 2 + sigN2 ** 2) ** 0.5

    # Draw from each posterior predictive distribution to simulate the matchup
    pts1 = round(random.normal(muN1, sigNpred1), 3)
    pts2 = round(random.normal(muN2, sigNpred2), 3)

    return pts1, pts2


def get_records_dict(teams, records):
    records_dict = {}
    for team in teams:
        query = f"Team == '{team}'"
        wins = records.query(query).W.reset_index(drop=True).loc[0]
        losses = records.query(query).L.reset_index(drop=True).loc[0]
        pts = records.query(query).pts.reset_index(drop=True).loc[0]
        records_dict[team] = [wins, losses, pts]
    return records_dict


def sim_season(data, schedule, team_dict, teams, records, bayes=False):
    df = get_team_points(data)
    v = data.dropna().week.max()
    records_dict = get_records_dict(teams=teams, records=records)
    for team in teams:
        query = f"Team == '{team}'"
        wins = records.query(query).W.reset_index(drop=True).loc[0]
        losses = records.query(query).L.reset_index(drop=True).loc[0]
        pts = records.query(query).pts.reset_index(drop=True).loc[0]
        records_dict[team] = [wins, losses, pts]
    for i in range(len(schedule)):
        team1 = schedule.loc[i].team1
        team2 = schedule.loc[i].team2

        if not bayes:
            pts1 = round(
                team_dict[team1][1] * random.standard_t(df=v) + team_dict[team1][0], 3
            )
            pts2 = round(
                team_dict[team2][1] * random.standard_t(df=v) + team_dict[team2][0], 3
            )
        else:
            pts1, pts2 = get_points(team1, team2, df=df)
        records_dict[team1][2] = round(pts1 + records_dict[team1][2], 2)
        records_dict[team2][2] = round(pts2 + records_dict[team2][2], 2)

        if pts1 > pts2:
            records_dict[team1][0] += 1
            records_dict[team2][1] += 1
        else:
            records_dict[team2][0] += 1
            records_dict[team1][1] += 1
    return records_dict


def seed_playoffs(records_dict, teams):
    wins, losses, pts = [], [], []

    for team in teams:
        wins.append(records_dict[team][0])
        losses.append(records_dict[team][1])
        pts.append(records_dict[team][2])
    standings = pd.DataFrame(
        {"team": teams, "wins": wins, "losses": losses, "points": pts}
    )

    standings = standings.sort_values(
        by=["wins", "points"], ascending=False
    ).reset_index(drop=True)
    return standings


def simulate_playoffs(data, standings, championship_dict, teams):
    team_dict = get_likelihood_dict(teams)
    v = data.dropna().week.max()

    seed1 = standings.team.loc[0]
    seed2 = standings.team.loc[1]
    seed3 = standings.team.loc[2]
    seed4 = standings.team.loc[3]

    championship = []
    consolation = []

    # Seed 1 v Seed 4
    pts1 = round(team_dict[seed1][1] * random.standard_t(df=v) + team_dict[seed1][0], 3)
    pts4 = round(team_dict[seed4][1] * random.standard_t(df=v) + team_dict[seed4][0], 3)

    if pts1 > pts4:
        championship.append(seed1)
        consolation.append(seed4)
    else:
        championship.append(seed4)
        consolation.append(seed1)
    # Seed 2 v Seed 3
    pts2 = round(team_dict[seed2][1] * random.standard_t(df=v) + team_dict[seed2][0], 3)
    pts3 = round(team_dict[seed3][1] * random.standard_t(df=v) + team_dict[seed3][0], 3)

    if pts2 > pts3:
        championship.append(seed2)
        consolation.append(seed3)
    else:
        championship.append(seed3)
        consolation.append(seed2)
    # Championship Game
    pts1 = round(
        team_dict[championship[0]][1] * random.standard_t(df=v)
        + team_dict[championship[0]][0],
        3,
    )
    pts2 = round(
        team_dict[championship[1]][1] * random.standard_t(df=v)
        + team_dict[championship[1]][0],
        3,
    )

    if pts1 > pts2:
        championship_dict[championship[0]][0] += 1
        championship_dict[championship[1]][1] += 1
    else:
        championship_dict[championship[1]][0] += 1
        championship_dict[championship[0]][1] += 1
    # Consolation Game
    pts1 = round(
        team_dict[consolation[0]][1] * random.standard_t(df=v)
        + team_dict[consolation[0]][0],
        3,
    )
    pts2 = round(
        team_dict[consolation[1]][1] * random.standard_t(df=v)
        + team_dict[consolation[1]][0],
        3,
    )

    if pts1 > pts2:
        championship_dict[consolation[0]][2] += 1
        championship_dict[consolation[1]][3] += 1
    else:
        championship_dict[consolation[1]][2] += 1
        championship_dict[consolation[0]][3] += 1
    return championship_dict


def simulate_season(iterations, bayes, teams, data, schedule, team_dict, records):
    wins_dict = {}
    for team in teams:
        wins_dict[team] = []
    championship_dict = {}
    for team in teams:
        championship_dict[team] = [0, 0, 0, 0]
    print("Iteration count: ", end="")
    for i in range(iterations):
        if i % 100 == 0:
            print(i, end=", ")
        records_dict = sim_season(
            data=data,
            schedule=schedule,
            team_dict=team_dict,
            teams=teams,
            records=records,
            bayes=bayes,
        )
        standings = seed_playoffs(records_dict=records_dict)
        champioship_dict = simulate_playoffs(
            data=data, standings=standings, championship_dict=championship_dict
        )

        for team in teams:
            wins_dict[team].append(records_dict[team][0])
    return championship_dict

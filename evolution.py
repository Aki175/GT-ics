# Name: Hakan Bektas & Akbar Ismatullayev
# Student ID: 15178684 & Add here (akbar)

import numpy as np
import heapq
import matplotlib.pyplot as plt
import itertools
from tkinter import Tk, Label, Entry, Button, Checkbutton, BooleanVar


# General notes: cooperate = 1, defect = 0.

class Strategy:
    """
    This class contains multiple strategies and the methods to let them
    play matches in a round-robin tournament.
    """
    def __init__(self):
        # Store the history for the "current" match.
        self.historyOwn = []
        self.historyOpp = []

        self.reward = 0
        self.temptation = 0
        self.sucker = 0
        self.punishment = 0
        self.genetic = False
        self.genetic_previous_n = 1
        self.population_size = 0


        self.stratList = [
        self.titForTat,
        self.equally_random,
        self.cRandom,
        self.dRandom,
        self.moreNaive,
        self.statisticalPlayer,
        self.WSLS,
        self.Sneaky_Temptation,
        ]

        np.random.seed(42)



    def clearHistory(self):
        """
        Clears the stored history for a new match.
        """
        self.historyOwn = []
        self.historyOpp = []

    # --- Strategies ---

    def titForTat(self):
        """
        Cooperate on the first move, then copy opponent's last move.
        """
        if len(self.historyOpp) == 0:
            return 1
        return self.historyOpp[-1]

    def equally_random(self):
        """
        Randomly choose 0 or 1 with 50% chance.
        """
        return np.random.randint(0, 2)

    def cRandom(self):
        """
        Random but more likely to cooperate (75%).
        """
        return np.random.choice([0, 1], p=[0.25, 0.75])

    def dRandom(self):
        """
        Random but more likely to defect (75%).
        """
        return np.random.choice([0, 1], p=[0.75, 0.25])

    def moreNaive(self):
        """
        After 2 consecutive betrayals from the opponent, we also betray.
        """
        if len(self.historyOpp) >= 2:
            if self.historyOpp[-1] == 0 and self.historyOpp[-2] == 0:
                return 0
        return 1

    def statisticalPlayer(self):
        """
        If the opponent has defected more than half the time so far,
        we defect. Otherwise we cooperate.
        """
        if len(self.historyOpp) == 0:
            return 1
        if self.historyOpp.count(0) > (len(self.historyOpp) / 2):
            return 0
        return 1

    # deze is good tegen deadlocks omdat we switchen by punishment
    def WSLS(self):
        '''
        win stay: meaning if the pay off is good thus both cooperated or we
        got the temptation pay off we say with our choice like before.
        If we lose shift, meaning if the pay off is bad thus we got punished
        for both deflecting or we got a suckers pay off we switch our last
        choice.
        '''
        if len(self.historyOpp) == 0:
            return 1
        lst_op = self.historyOpp[-1]
        lst_own = self.historyOwn[-1]

        if (lst_own == 1 and lst_op == 1) or (lst_own == 0 and lst_op == 1):
            return lst_own

        # If we lose we switch
        return (lst_own + 1) % 2

    def Sneaky_Temptation(self):
        '''
        always cooperate but every third round deflect
        '''

        if len(self.historyOpp) == 0:
            return 1
        if (len(self.historyOwn) + 1) % 3 == 0:
            return 0

        return 1

    def Anti_Tit_For_Tat(self):
        '''
        reverse of tit for that
        '''
        if len(self.historyOpp) == 0:
            return 1
        return int((self.historyOpp[-1] or 0) + 1) % 2


    def Grim(self):
        '''
        always cooperate but when oppenent ever defected than always defect.
        it is not forgiving
        '''
        if len(self.historyOpp) == 0:
            return 1

        if 0 in self.historyOpp:
            return 0

        return 1

    def Adaptive_Gradual_TFT(self):
        '''
        It always cooperates only after the oppenent deflect 2 times
        than it deflect always since it has a grudge but it is forgiven
        when the opponent cooperates till the revenge counter is below 2.
        '''

        if len(self.historyOpp) == 0:
            self.revenge_counter = 0
            return 1

        if self.historyOpp[-1] == 0:
            self.revenge_counter += 1
        else:
            self.revenge_counter = max (0, self.revenge_counter - 1)

        if self.revenge_counter >= 2:
            return 0


        return 1

    def random_strat(self, rule_table):

        """
        Does the random strategies based on their rule table. We look at the
        history till n and make the combinations of the history of our own
        and our opp. This is our key in our dictonary.
        """
        n = self.genetic_previous_n

        if len(self.historyOpp) < n:
            response = self.equally_random()
            print(response)
            return response

        own_moves = self.historyOwn[-n:]
        opp_moves = self.historyOpp[-n:]
        combination_key = tuple(zip(own_moves, opp_moves))
        print(f"Combination key: {combination_key}")
        print(rule_table[combination_key])
        return rule_table[combination_key]


    # Match / Round logic

    # is dit de payoff table?
    def scoreRound(self, a, b):
        """
        Returns (pointsA, pointsB) for a single round,
        given the moves a, b in {0,1}.
        """
        if a == 1 and b == 1:
            return (self.reward, self.reward)
        elif a == 0 and b == 0:
            return (self.punishment, self.punishment)
        elif a == 1 and b == 0:
            return (self.sucker, self.temptation)
        else:  # a==0, b==1
            return (self.temptation, self.sucker)


    def playMatch(self, stratA, stratB, rounds=10):
        """
        Plays rounds of two strategies.
        Returns final (scoreA, scoreB).
        """
        # Clear histories for each new match
        self.clearHistory()
        histA_own = self.historyOwn
        histA_opp = self.historyOpp

        histB_own = []
        histB_opp = []

        # we moeten hier adden van of is aangekruist of niet.

        scoreA, scoreB = 0, 0


        for _ in range(rounds):
            if self.genetic:
                moveA = self.random_strat(stratA)
            else:
                print("stratA" + str(stratA))
                moveA = stratA()


            # A's move
            # record A's move in A's own history
            histA_own.append(moveA)

            # B's move
            moveB = stratB()
            histB_own.append(moveB)

            # Now update each other's opponent history
            histA_opp.append(moveB)
            histB_opp.append(moveA)

            # Score them
            ptsA, ptsB = self.scoreRound(moveA, moveB)
            scoreA += ptsA
            scoreB += ptsB

        return scoreA, scoreB

    def generate_random_rule_table(self):
        # since for every n you have 4 states
        choices = [(0, 0), (0, 1), (1, 0), (1, 1)]
        # responses
        states = [0, 1]
        all_combinations = list(itertools.product(choices,
                                                  repeat=self.genetic_previous_n))
        print(all_combinations)

        rule_table = {combination: np.random.choice(states) for combination
                      in all_combinations}
        print(rule_table)


        return rule_table

    def tournament_random(self, rounds=10):

        self.sumScores = [0] * self.population_size
        self.cooperation_count = [0] * self.population_size
        self.retaliating_count = [0] * self.population_size

        rule_tables = [self.generate_random_rule_table()
                       for _ in range(self.population_size)]

        # Every rule table plays against every strategy
        for rt in range(len(rule_tables)):
            rule_table = rule_tables[rt]
            print(f"Rule table {rt}: {rule_table}")

            for s in range(len(self.stratList)):
                strategy = self.stratList[s]
                print(f"Strategy {s}: {strategy}")

                scoreA, scoreB = self.playMatch(rule_table, strategy, rounds)

                self.sumScores[rt] += scoreB
                # Cooperation.
                if scoreA == scoreB:
                    self.cooperation_count[rt] += 1
                    # Adventageous defection.
                elif scoreB > scoreA:
                    self.retaliating_count[rt] += 1

    def plot_rule_table(self):
        _, ax = plt.subplots(1, 1, figsize=(10, 6))
        x = np.arange(self.population_size)
        width = 0.32

        # Plot for sume scores
        _ = ax.bar(x - width / 2, self.sumScores, width, label='Sum Scores',
        color='blue')
        plt.tight_layout()
        plt.show()


    def tournament(self, rounds=10):
        """
        Runs a round-robin among all included strategies.
        Collects per-strategy lowest, highest, average, totalWins, mutual.
        Then prints results and uses a heap to pick top-5 by mutual points.
        """
        # Store references to the strategy methods in a list:
        if self.genetic:
            self.tournament_random()

        self.names = [
            "Tit-for-Tat",
            "Equally Random",
            "C-Random",
            "D-Random",
            "More Naive",
            "Statistical Player",
            "WSLS",
            "Sneaky_Temptation",
        ]

        n = len(self.stratList)
        print(n)
        print(len(self.names))

        self.lowest = [9999999] * n
        self.highest = [0] * n
        self.sumScores = [0] * n
        self.wins = [0] * n
        self.mutual = [0] * n
        self.gamesPlayed = [0] * n
        self.lowest_mutual = [9999999] * n
        self.highest_mutual = [0] * n
        self.mutual_scores_per_strategy = [[] for _ in range(n)]

        # Let every strategy play with each other

        # stel we hebben 10 combinaties begint bij i 1, j is 2 till n.
        # i heeft all gevochten dus nu i is 2 j moet altijd plus 1 omdat je
        # anders dubbel tegen gaat.
        for i in range(n):
            for j in range(i + 1, n):
                scoreI, scoreJ = self.playMatch(self.stratList[i],
                                                 self.stratList[j], rounds)
                mutual_score = scoreI + scoreJ

                # Update mutual scores per strategy
                self.mutual_scores_per_strategy[i].append(mutual_score)
                self.mutual_scores_per_strategy[j].append(mutual_score)

                # Update i
                if scoreI < self.lowest[i]:
                    self.lowest[i] = scoreI
                if scoreI > self.highest[i]:
                    self.highest[i] = scoreI
                self.sumScores[i] += scoreI
                self.gamesPlayed[i] += 1
                self.mutual[i] += mutual_score

                # Update j
                if scoreJ < self.lowest[j]:
                    self.lowest[j] = scoreJ
                if scoreJ > self.highest[j]:
                    self.highest[j] = scoreJ
                self.sumScores[j] += scoreJ
                self.gamesPlayed[j] += 1
                self.mutual[j] += mutual_score

                # Update lowest and highest mutual scores
                if mutual_score < self.lowest_mutual[i]:
                    self.lowest_mutual[i] = mutual_score
                if mutual_score > self.highest_mutual[i]:
                    self.highest_mutual[i] = mutual_score
                if mutual_score < self.lowest_mutual[j]:
                    self.lowest_mutual[j] = mutual_score
                if mutual_score > self.highest_mutual[j]:
                    self.highest_mutual[j] = mutual_score

                # Who wins
                if scoreI > scoreJ:
                    self.wins[i] += 1
                elif scoreJ > scoreI:
                    self.wins[j] += 1
                else:
                    pass  # tie, do nothing

        # Compute average scores
        self.avg = [0] * n
        for i in range(n):
            if self.gamesPlayed[i] > 0:
                self.avg[i] = self.sumScores[i] / self.gamesPlayed[i]
            else:
                # Never played, set everything to 0
                self.lowest[i] = 0
                self.highest[i] = 0
                self.avg[i] = 0

        # Compute mutual average per strategy
        self.mutual_avg = [0] * n
        for i in range(n):
            if len(self.mutual_scores_per_strategy[i]) > 0:
                self.mutual_avg[i] = np.mean(self.mutual_scores_per_strategy[i])
            else:
                self.mutual_avg[i] = 0

        print("===== Tournament Results =====")
        for i in range(n):
            print(f"{self.names[i]}:")
            print(f"  Lowest = {self.lowest[i]}")
            print(f"  Highest= {self.highest[i]}")
            print(f"  Average= {self.avg[i]:.2f}")
            print(f"  Wins   = {self.wins[i]}")
            print(f"  Mutual = {self.mutual[i]}")
            print(f"  Mutual Avg = {self.mutual_avg[i]:.2f}")
            print("")

        # Use of heap to retrieve and put the highest value efficiently
        pq = []
        for i in range(n):
            # Negative so the largest mutual is at pop()
            heapq.heappush(pq, (-self.mutual[i], i))

        print("===== TOP-5 By Mutual Points =====")
        topCount = min(5, n)
        for rank in range(topCount):
            best = heapq.heappop(pq)
            idx = best[1]
            actualMut = -best[0]
            print(f"{rank + 1}. {self.names[idx]} with mutual={actualMut}")

    def plot(self):
        """
        Plots the  results of the tournament. It will use a bar chart to show the
        minimal, maximal, and average scores of each strategy.
        """
        # Prepare the data for the plot
        data = list(zip(self.names, self.avg, self.lowest, self.highest))
        # Sort the data by average score
        data.sort(key=lambda x: x[1], reverse=True)
        # Unpack the sorted data
        self.names, self.avg, self.lowest, self.highest = zip(*data)

        # Create a figure with two subplots
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 6))
        # Plot for minimal, maximal, and average scores
        x = np.arange(len(self.names))
        width = 0.32

        # Average scores
        _ = ax1.bar(x - width / 2, self.avg, width, label='Average', color='blue')
        # Error bars for minimal and maximal scores
        ax1.errorbar(x - width / 2, self.avg,
                    yerr=[np.subtract(self.avg, self.lowest),
                        np.subtract(self.highest, self.avg)],
                    fmt='o', color='red', label='Min/Max score', capsize=5)

        # Labels and title for the first plot
        ax1.set_xlabel('Strategies')
        ax1.set_ylabel('Scores')
        ax1.set_title('Minimal, maximal, and average scores per strategy')
        ax1.set_xticks(x - width / 2)
        ax1.set_xticklabels(self.names, rotation=45)
        ax1.legend()

        # Prepare data for mutual scores plot
        # Create a list of strategy pairs and their mutual scores
        strategy_pairs = []
        mutual_scores = []
        mutual_lowest = []
        mutual_highest = []

        for i in range(len(self.names)):
            for j in range(i + 1, len(self.names)):
                strategy_pairs.append(f"{self.names[i]} vs {self.names[j]}")
                mutual_score = self.mutual_scores_per_strategy[i][j - i - 1]
                mutual_scores.append(mutual_score)
                mutual_lowest.append(self.lowest_mutual[i])
                mutual_highest.append(self.highest_mutual[i])

        # Combine into a list of tuples and sort by mutual score (descending)
        mutual_data = list(zip(strategy_pairs, mutual_scores, mutual_lowest, mutual_highest))
        mutual_data.sort(key=lambda x: x[1], reverse=True)

        # Select the top 5
        top_5_mutual = mutual_data[:5]

        # Unpack the top 5
        top_strategy_pairs, top_mutual_scores, top_mutual_lowest, top_mutual_highest = zip(*top_5_mutual)

        # Plot for mutual scores (top 5 only)
        x_mutual = np.arange(len(top_strategy_pairs))  # x-axis positions for
        # mutual scores
        _ = ax2.bar(x_mutual, top_mutual_scores, width, label='Mutual Score',
                    color='green')
        # Error bars for minimal and maximal mutual scores
        ax2.errorbar(x_mutual, top_mutual_scores,
                    yerr=[np.subtract(top_mutual_scores, top_mutual_lowest),
                        np.subtract(top_mutual_highest, top_mutual_scores)],
                    fmt='o', color='orange', label='Min/Max Mutual Score',
                    capsize=5)

        print(top_strategy_pairs)
        print(top_mutual_scores)

        # Labels and title for the second plot
        ax2.set_xlabel('Strategy Pairs')
        ax2.set_ylabel('Mutual Scores')
        ax2.set_title(
            'Top 5 Mutual scores per strategy pair\n'
            'Mutual score = sum of scores over all rounds\n'
            'Min/Max = lowest/highest mutual score in a single match'
        )
        ax2.set_xticks(x_mutual)
        ax2.set_xticklabels(top_strategy_pairs, rotation=45)
        ax2.legend()

        # Plot for average wins per strategy
        avg_wins = [self.wins[i] / self.gamesPlayed[i]
                    if self.gamesPlayed[i] > 0 else 0
                    for i in range(len(self.names))]

        #Combine into a list of tuples and sort by mutual score (descending)
        win_data = list(zip(self.names, avg_wins))
        win_data.sort(key=lambda x: x[1], reverse=True)

        # Unpack the top 5
        self.names, avg_wins = zip(*win_data)

        _ = ax3.bar(x, avg_wins, width, label='Average Wins', color='purple')

        # Labels and title for the third plot
        ax3.set_xlabel('Strategies')
        ax3.set_ylabel('Average Wins')
        ax3.set_title('Average number of wins per strategy')
        ax3.set_xticks(x)
        ax3.set_xticklabels(self.names, rotation=45)
        ax3.legend()

        plt.tight_layout()
        plt.show()

class SimpleGUI:
    """
    GUI with a 'Start Tournament' button that calls Strategy().tournament().
    """
    def __init__(self):
        self.root = Tk()
        self.root.title("Tournament GUI")

        Label(self.root, text="Number of rounds per match:").grid(
            row=0, column=0, padx=10, pady=5
        )
        self.roundsEntry = Entry(self.root)
        self.roundsEntry.insert(0, "10")
        self.roundsEntry.grid(row=0, column=1, padx=10, pady=5)

        Label(self.root, text="Reward (C, C):").grid(row=1, column=0, padx=10, pady=5)
        self.rewardEntry = Entry(self.root)
        self.rewardEntry.insert(0, "3")
        self.rewardEntry.grid(row=1, column=1, padx=10, pady=5)

        Label(self.root, text="Temptation (D, C):").grid(row=2, column=0, padx=10, pady=5)
        self.temptationEntry = Entry(self.root)
        self.temptationEntry.insert(0, "5")
        self.temptationEntry.grid(row=2, column=1, padx=10, pady=5)

        Label(self.root, text="Sucker (C, D):").grid(row=3, column=0, padx=10, pady=5)
        self.suckerEntry = Entry(self.root)
        self.suckerEntry.insert(0, "0")
        self.suckerEntry.grid(row=3, column=1, padx=10, pady=5)

        Label(self.root, text="Punishment (D, D):").grid(row=4, column=0, padx=10, pady=5)
        self.punishmentEntry = Entry(self.root)
        self.punishmentEntry.insert(0, "1")
        self.punishmentEntry.grid(row=4, column=1, padx=10, pady=5)


        Label(self.root, text="Genetic Mode:").grid(row=5, column=0, padx=10, pady=5)
        self.geneticModeVar = BooleanVar()
        self.geneticModeCheck = Checkbutton(self.root, variable=self.geneticModeVar)
        self.geneticModeCheck.grid(row=5, column=1, padx=10, pady=5)

        Label(self.root, text="Previous Moves (N):").grid(row=6, column=0, padx=10, pady=5)
        self.nEntry = Entry(self.root)
        self.nEntry.insert(0, "1")  # Default N=1
        self.nEntry.grid(row=6, column=1, padx=10, pady=5)

        Label(self.root, text="Population size").grid(row=7, column=0, padx=10, pady=5)
        self.PopSize = Entry(self.root)
        self.PopSize.insert(0, "20")
        self.PopSize.grid(row=7, column=1, padx=10, pady=5)



        # Button to start the tournament
        self.tourneyButton = Button(self.root, text="Start Tournament",
                                    command=self.startTournament)
        self.tourneyButton.grid(row=8, column=0, columnspan=2, pady=10)

    def startTournament(self):
        """
        Reads 'rounds' from the GUI, then runs the Strategy.tournament().
        """
        s = Strategy()

        s.reward = int(self.rewardEntry.get())
        s.temptation = int(self.temptationEntry.get())
        s.sucker = int(self.suckerEntry.get())
        s.punishment = int(self.punishmentEntry.get())

        s.genetic = self.geneticModeVar.get()
        s.genetic_previous_n = int(self.nEntry.get())

        s.population_size = int(self.PopSize.get())


        r = int(self.roundsEntry.get())
        # s.generate_random_rule_table()
        s.tournament_random(rounds=r)

        if s.genetic:
            s.plot_rule_table()
        else:
            s.plot()
        # s.tournament_random(rounds=r)
        # s.tournament(rounds=r)
        # s.plot()

    def start(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = SimpleGUI()
    gui.start()
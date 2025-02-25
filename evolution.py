# Name: Hakan Bektas & Akbar Ismatullayev
# Student ID: 15178684 & Add here (akbar)

import numpy as np
import heapq
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Entry, Button, Checkbutton, BooleanVar
# from pyics import Model


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

    def equallyRandom(self):
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

    # Akbar voeg hier zelf nog 5 strategieen toe.

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


    # Match / Round logic

    # is dit de payoff table?
    def scoreRound(self, a, b):
        """
        Returns (pointsA, pointsB) for a single round,
        given the moves a, b in {0,1}.
        """
        if a == 1 and b == 1:
            return (3, 3)
        elif a == 0 and b == 0:
            return (1, 1)
        elif a == 1 and b == 0:
            return (0, 5)
        else:  # a==0, b==1
            return (5, 0)

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

        scoreA, scoreB = 0, 0

        for _ in range(rounds):
            # A's move
            moveA = stratA()
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


    def tournament(self, rounds=10):
        """
        Runs a round-robin among all included strategies.
        Collects per-strategy lowest, highest, average, totalWins, mutual.
        Then prints results and uses a heap to pick top-5 by mutual points.
        """
        # Store references to the strategy methods in a list:
        stratList = [
            self.titForTat,
            self.equallyRandom,
            self.cRandom,
            self.dRandom,
            self.moreNaive,
            self.statisticalPlayer,
            self.WSLS,
            self.Sneaky_Temptation,
            self.Anti_Tit_For_Tat,
            self.Grim,
            self.Adaptive_Gradual_TFT
        ]

        self.names = [
            "Tit-for-Tat",
            "Equally Random",
            "C-Random",
            "D-Random",
            "More Naive",
            "Statistical Player"
            "WSLS",
            "Sneaky_Temptation",
            "Anti_Tit_For_Tat",
            "Grim",
            "Adaptive_Gradual_TFT"
        ]

        n = len(stratList)

        self.lowest = [9999999] * n
        self.highest = [0] * n
        self.sumScores = [0] * n
        self.wins = [0] * n
        self.mutual = [0] * n
        self.gamesPlayed = [0] * n
        self.lowest_mutual = [0] * n

        # Let every strategy play with each other
        for i in range(n):
            for j in range(i + 1, n):
                scoreI, scoreJ = self.playMatch(stratList[i], stratList[j],
                                                rounds)
                # update i
                if scoreI < self.lowest[i]:
                    self.lowest[i] = scoreI
                if scoreI > self.highest[i]:
                    self.highest[i] = scoreI
                self.sumScores[i] += scoreI
                self.gamesPlayed[i] += 1
                self.mutual[i] += (scoreI + scoreJ)
                # lowest for mutual
                if self.mutual[i] < (scoreI + scoreJ):
                    self.lowest_mutual[i] = (scoreI + scoreJ)

                # update j
                if scoreJ < self.lowest[j]:
                    self.lowest[j] = scoreJ
                if scoreJ > self.highest[j]:
                    self.highest[j] = scoreJ
                self.sumScores[j] += scoreJ
                self.gamesPlayed[j] += 1
                self.mutual[j] += (scoreI + scoreJ)
                if self.mutual[j] < (scoreI + scoreJ):
                    self.lowest_mutual[j] = (scoreI + scoreJ)

                # who wins
                if scoreI > scoreJ:
                    self.wins[i] += 1
                elif scoreJ > scoreI:
                    self.wins[j] += 1
                else:
                    pass  # tie, do nothing

        # compute average
        self.avg = [0] * n
        for i in range(n):
            if self.gamesPlayed[i] > 0:
                self.avg[i] = self.sumScores[i] / self.gamesPlayed[i]
            else:
                # never played, set everything to 0
                self.lowest[i] = 0
                self.highest[i] = 0
                self.avg[i] = 0

        # Compute mutual average, lowest, and highest
        self.mutual_avg = np.mean(self.mutual_scores) if self.mutual_scores else 0
        self.mutual_lowest = np.min(self.mutual_scores) if self.mutual_scores else 0
        self.mutual_highest = np.max(self.mutual_scores) if self.mutual_scores else 0

        print("===== Tournament Results =====")
        for i in range(n):
            print(f"{self.names[i]}:")
            print(f"  Lowest = {self.lowest[i]}")
            print(f"  Highest= {self.highest[i]}")
            print(f"  Average= {self.avg[i]:.2f}")
            print(f"  Wins   = {self.wins[i]}")
            print(f"  Mutual = {self.mutual[i]}")
            print("")

        # Use of heap to retrieve and put the highest value efficiently
        pq = []
        for i in range(n):
            # negative so the largest mutual is at pop()
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
        Plots the results of the tournament. It will use a bar chart to show the
        minimal, maximal, and average scores of each strategy.
        """
        # Prepare the data for the plot
        data = list(zip(self.names, self.avg, self.lowest, self.highest))
        # Sort the data by average score
        data.sort(key=lambda x: x[1], reverse=True)
        # Unpack the sorted data
        self.names, self.avg, self.lowest, self.highest = zip(*data)

        # Create a figure with a two subplot
        _, ax = plt.subplots(figsize=(10, 6))
        # _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))


        # Plot for minimal, maximal, and average scores
        x = np.arange(len(self.names))  # x-axis positions
        width = 0.35  # Width of the bars

        # Average scores
        _ = ax.bar(x - width / 2, self.avg, width, label='Average', color='blue')
        # Error bars for minimal and maximal scores
        ax.errorbar(x - width / 2, self.avg,
                    yerr=[np.subtract(self.avg, self.lowest),
                          np.subtract(self.highest, self.avg)],
                    fmt='o', color='red', label='Min/Max score', capsize=5)

        # Labels and title for the plot
        ax.set_xlabel('Strategies')
        ax.set_ylabel('Scores')
        ax.set_title('Minimal, maximal, and average scores per strategy')
        ax.set_xticks(x - width / 2)
        ax.set_xticklabels(self.names, rotation=45)
        ax.legend()

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plot
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

        # Button to start the tournament
        self.tourneyButton = Button(self.root, text="Start Tournament",
                                    command=self.startTournament)
        self.tourneyButton.grid(row=1, column=0, columnspan=2, pady=10)

    def startTournament(self):
        """
        Reads 'rounds' from the GUI, then runs the Strategy.tournament().
        """
        s = Strategy()
        r = int(self.roundsEntry.get())
        s.tournament(rounds=r)
        s.plot()

    def start(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = SimpleGUI()
    gui.start()
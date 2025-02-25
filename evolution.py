# Name: Hakan Bektas & Akbar Islamov
# Student ID: 15178684 & Add here (akbar)

import numpy as np
import heapq
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Entry, Button, Checkbutton, BooleanVar
from pyics import Model


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


    # Match / Round logic

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
            moveB = None
            if stratB == self.titForTat:
                if len(histB_opp) == 0:
                    moveB = 1
                else:
                    moveB = histB_opp[-1]
            elif stratB == self.equallyRandom:
                moveB = np.random.randint(0, 2)
            elif stratB == self.cRandom:
                moveB = np.random.choice([0, 1], p=[0.25, 0.75])
            elif stratB == self.dRandom:
                moveB = np.random.choice([0, 1], p=[0.75, 0.25])
            elif stratB == self.moreNaive:
                if len(histB_opp) >= 2:
                    if histB_opp[-1] == 0 and histB_opp[-2] == 0:
                        moveB = 0
                    else:
                        moveB = 1
                else:
                    moveB = 1
            elif stratB == self.statisticalPlayer:
                if len(histB_opp) == 0:
                    moveB = 1
                else:
                    if histB_opp.count(0) > len(histB_opp)/2:
                        moveB = 0
                    else:
                        moveB = 1
            else:
                pass  # Akbar voeg hier je eigen strategien toe

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
            self.statisticalPlayer
            # Akbar voeg hier je eigen strategien toe
        ]

        names = [
            "Tit-for-Tat",
            "Equally Random",
            "C-Random",
            "D-Random",
            "More Naive",
            "Statistical Player"
            # Akbar voeg hier je strategie namen toe
        ]

        n = len(stratList)

        lowest = [9999999]*n
        highest = [0]*n
        sumScores = [0]*n
        wins = [0]*n
        mutual = [0]*n
        gamesPlayed = [0]*n

        # Let every strategy play with each other (e.g there are 5 strategies
        # called 1 2 3 4 5. Then 1 plays against 2 3 4 5 after 2 plays against
        # 3 4 5 and so on).
        for i in range(n):
            for j in range(i+1, n):
                scoreI, scoreJ = self.playMatch(stratList[i], stratList[j],
                                                rounds)
                # update i
                if scoreI < lowest[i]:
                    lowest[i] = scoreI
                if scoreI > highest[i]:
                    highest[i] = scoreI
                sumScores[i] += scoreI
                gamesPlayed[i] += 1
                mutual[i] += (scoreI + scoreJ)

                # update j
                if scoreJ < lowest[j]:
                    lowest[j] = scoreJ
                if scoreJ > highest[j]:
                    highest[j] = scoreJ
                sumScores[j] += scoreJ
                gamesPlayed[j] += 1
                mutual[j] += (scoreI + scoreJ)

                # who wins
                if scoreI > scoreJ:
                    wins[i] += 1
                elif scoreJ > scoreI:
                    wins[j] += 1
                else:
                    pass  # tie, do nothing

        # compute average
        avg = [0]*n
        for i in range(n):
            if gamesPlayed[i] > 0:
                avg[i] = sumScores[i] / gamesPlayed[i]
            else:
                # never played, set everything to 0
                lowest[i] = 0
                highest[i] = 0
                avg[i] = 0

        print("===== Tournament Results =====")
        for i in range(n):
            print(f"{names[i]}:")
            print(f"  Lowest = {lowest[i]}")
            print(f"  Highest= {highest[i]}")
            print(f"  Average= {avg[i]:.2f}")
            print(f"  Wins   = {wins[i]}")
            print(f"  Mutual = {mutual[i]}")
            print("")

        # Use of heap to retrieve and put the highest value efficiently (Good
        # BIG o).
        pq = []
        for i in range(n):
            # negative so the largest mutual is at pop()
            heapq.heappush(pq, (-mutual[i], i))

        print("===== TOP-5 By Mutual Points =====")
        topCount = min(5, n)
        for rank in range(topCount):
            best = heapq.heappop(pq)
            idx = best[1]
            actualMut = -best[0]
            print(f"{rank+1}. {names[idx]} with mutual={actualMut}")

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

    def start(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = SimpleGUI()
    gui.start()



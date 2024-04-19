class Cellule:
    #ajouter class robot et drone qui ne sont pas reliés à la classe cellule
    def __init__(self, arbre=False, obstacle=None, date=None, déchet=False):
        self.arbre = arbre
        self.obstacle = obstacle
        self.date = date
        self.déchet = déchet
        

def créer_tableau_2D(rows, cols):
    tableau = [[Cellule() for _ in range(cols)] for _ in range(rows)]
    return tableau


if __name__ == "__main__":
    
    tableau_2D = créer_tableau_2D(5, 5)
    tableau_2D[0][0].arbre = True
    tableau_2D[2][3].obstacle = "tronc"
    tableau_2D[4][2].date = (2024, 4, 10) # a changer en heure à incrémenter 
    tableau_2D[1][1].déchet = True

    for row in tableau_2D:
        for cellule in row:
            print(f"Arbre: {cellule.arbre}, Obstacle: {cellule.obstacle}, Date: {cellule.date}, Déchet: {cellule.déchet}")
        print()
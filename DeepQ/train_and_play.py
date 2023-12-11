from pacman import layout, readCommand, runGames
import pacmanDQN_Agents  # Import your DQN agent module

def train_and_play():
    # Training configuration
    train_args = readCommand(["-p", "PacmanDQN", "-n", "100", "-x", "100", "-l", "smallGrid"])
    train_args['numTraining'] = 100
    train_args['numGames'] = 110

    # Training
    print("Training...")
    runGames(**train_args)

    # Playing configuration
    play_args = readCommand(["-p", "PacmanDQN", "-n", "10", "-x", "10", "-l", "smallGrid"])

    # Playing
    print("Playing...")
    runGames(**play_args)

if __name__ == "__main__":
    train_and_play()

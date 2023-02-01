Thank you for reading this guide! 

To compile the host and client code, please enter the following commands:
"gcc TicTacToe_Host.c tictactoe.c -o TicTacToe_Host -Wall"
"gcc TicTacToe_Client.c tictactoe.c -o TicTacToe_Client -Wall"

This will create the two executables required to play the game.

Before you start playing, you will need two terminal windows.
To start the game, one window must enter the command ".\TicTacToe_Host <grid_size>" where grid_size is an integer in the range [3,10]. 
The second window must enter the command ".\TicTacToe_Client".

Upon starting the software properly, the host instance and the client instance should both have the initial Tic Tac Toe grid printed to screen.
Players take turns playing the game by typing in the number corresponding to the square they would like to select. 

At any time, the game can be interrupted and ended by either instance typing "exit" on their turn. 
Otherwise, the game will commence until a winner is announced or a draw is reached. 

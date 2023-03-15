I included the standalone tool in this so that you can play with the visualization as well as watch it in action in my presentation. I personally think it's quite fun to play with and make maps where the two algorithms "race" each other. 

Because this software was made in Unity, I can't upload the entire Unity project because it is too large. What I have done instead is I made a SourceCode\ folder that contains each one of the C# programs I made as a part of this project. They won't really do anything on their own, since they belong inside of the Unity ecosystem and require Unity to actually make them do anything, but I figured that at least including the code I wrote even if it doesn't do anything would be sufficient enough to prove that it's my code. 

There are two classes at the high level directory, CameraController and PathfindingManager. CameraController allows the user to move and zoom the camera during the visualization, and PathfindingManager is the driver class that actually fires off the A* and BFS pathfinding algorithms, as well as allows users to interact with the grid by placing the start, finish, or any walls. 

Any code in the UI/ folder provides the functionality, menus, and scene loading that the UI needs.

The Grid/ folder contains the Grid class, that accepts generic TGridObject objects as elements. The grid consists of a two dimensional array of TGridObjects.

The AStar/ and BFS/ folders both contain _Pathfiding, _PathfindingDebugStepVisual, _PathfindingVisual, _PathNode, and _Testing each. The class that implements the actual pathfinding algorithm is _Pathfinding, and the two Visual classes are used to actually make the visualization. The _PathNode classes are the TGridObject passed to the grid class and store the variables involved in calculating the path in both algorithms.
 
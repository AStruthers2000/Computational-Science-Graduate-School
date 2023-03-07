(* ::Package:: *)

(* ::Title:: *)
(*CS 545: Data & Information Visualization - Group Final Project*)
(*Visualizing Interactions on StackOverflow*)


(* ::Subtitle:: *)
(*Authors: Nathan Chapman, Nick Haviland, Gihane Ndjeuha, Andrew Struthers, Kollin Trujillo*)


(* ::Section:: *)
(*Set up and Data Parsing*)


(* ::Input::Initialization:: *)
SetOptions[Rasterize,ImageSize->Large,ImageResolution->144]


(* ::Subsection:: *)
(*Directory Set up*)


(* ::Input::Initialization:: *)
Clear[dataPath,visualizationPath]
SetDirectory@NotebookDirectory[];
dataPath="data\\";
visualizationPath="visualizations\\";


(* ::Subsection:: *)
(*Data Parsing*)


(* ::Input::Initialization:: *)
Clear[fileName,fileStream,data]
fileName=dataPath<>"sx-stackoverflow-a2q.txt";
data=ReadList[fileStream=OpenRead@fileName,Number,3*(*NUMBER OF INTERACTIONS*)10^3,RecordLists->True];
Close@fileStream;
Dimensions@data


(* ::Subsection:: *)
(*Graph Variables*)


(* ::Input::Initialization:: *)
Clear[edgeList,vertexList,timeList,vertexDegreeListNormalized]
edgeList=DirectedEdge@@@data[[;;, ;;2]];
vertexList=VertexList@edgeList;
timeList=FromUnixTime/@data[[;;,3]];
vertexDegreeListNormalized=(#/Max@#)&@VertexDegree[edgeList]//N;


(* ::Section:: *)
(*Network Visualization*)


(* ::Subsection:: *)
(*Time Independent*)


(* ::Subsubsection:: *)
(*Degree Coloring*)


(* ::Input:: *)
(*BinCounts[vertexDegreeListNormalized,0.1]*)


(* ::Input:: *)
(*Clear[cf,maxVertexOutDegree]*)
(*maxVertexOutDegree=Max@VertexOutDegree[edgeList];*)
(*cf=ColorData[{"TemperatureMap",VertexOutDegree[edgeList]/maxVertexOutDegree//{Min@#,0.3Max@#}&}];*)


(* ::Input:: *)
(*Histogram[VertexOutDegree[edgeList]//N,AxesLabel->{"\nAnswers","Users"},ImagePadding->{{20,55},{20,20}}]*)
(*(*Export["Answer_Distribution.png",Rasterize@%]*)*)


(* ::Input:: *)
(*Max@VertexOutDegree[edgeList]*)


(* ::Input:: *)
(*GraphPlot[*)
(*edgeList,*)
(*VertexStyle->{*)
(*v_/;VertexOutDegree[edgeList,v]==0->Black,*)
(*v_/;VertexOutDegree[edgeList,v]>0:>cf[VertexOutDegree[edgeList,v]/maxVertexOutDegree]*)
(*},*)
(*VertexSize->1.75,*)
(*EdgeStyle->Green,*)
(*Background->Gray*)
(*];*)
(*Legended[*)
(*%,*)
(*Placed[*)
(*BarLegend[*)
(*{cf,VertexOutDegree[edgeList]/maxVertexOutDegree//DeleteDuplicates//{RankedMin[#,2],0.3Max@#}&},*)
(*LegendLabel->Placed["Normalized Answer #",Left],*)
(*LabelStyle->Black,*)
(*LegendLayout->"Row",*)
(*Background->Gray*)
(*],*)
(*Below*)
(*]*)
(*]*)
(*SystemOpen@Export["Degree_Graph.png",Rasterize[%,Background->Gray]]*)


(* ::Input:: *)
(*(*Users with the closest number of questions and answers*)*)
(*MinimalBy[vertexList,Abs[VertexInDegree[edgeList,#]-VertexOutDegree[edgeList,#]]&]*)
(*Subgraph[edgeList,%]*)


(* ::Input:: *)
(*(*Users with the most answers*)*)
(*MaximalBy[vertexList,VertexOutDegree[edgeList,#]&]*)
(*VertexOutDegree[edgeList,First@%]*)


(* ::Subsubsection:: *)
(*Individual User Networks*)


(* ::Input:: *)
(*Graph[#,GraphLayout->"StarEmbedding"]&/@Reverse@SortBy[GatherBy[edgeList,First],Length]*)


(* ::Subsubsection:: *)
(*Communities*)


(* ::Input:: *)
(*Clear@communityPlot*)
(*communityPlot=CommunityGraphPlot[edgeList,PlotLegends->Automatic,CommunityLabels->Range[11]]*)


(* ::Subsubsection:: *)
(*Cycles*)


(* ::Input:: *)
(*Clear@cyclePlot*)
(*cyclePlot=With[*)
(*{userCycles=FindCycle@edgeList},*)
(*GraphPlot[*)
(*HighlightGraph[*)
(*edgeList,*)
(*Style[#,Thickness[0.01],Red]&/@userCycles*)
(*],*)
(*ImageSize->Large,*)
(*PlotLabel->"Cycle between users "<>ToString[*)
(*VertexList/@userCycles//Row[#,", "]&*)
(*]*)
(*]*)
(*]*)


(* ::Subsubsection:: *)
(*Most Connected User Neighborhood*)


(* ::Input:: *)
(*Clear@mostConnectedUser*)
(*mostConnectedUser=Pick[vertexList,vertexDegreeListNormalized==Max@vertexDegreeListNormalized//Thread];*)


(* ::Input:: *)
(*Clear@mostConnectedUserPlot*)
(*mostConnectedUserPlot=GraphPlot[*)
(*HighlightGraph[edgeList,NeighborhoodGraph[edgeList,mostConnectedUser]],*)
(*PlotLabel->"The most connected user is: "<>ToString@First@mostConnectedUser*)
(*];*)


(* ::Subsubsection:: *)
(*Connected Components*)


(* ::Input:: *)
(*Clear[nonTrivialComponentList,nonTrivialComponentListPlot]*)
(*nonTrivialComponentList=Select[ConnectedComponents@edgeList,Length@#>1&];*)
(*nonTrivialComponentListPlot=GraphPlot[*)
(*HighlightGraph[edgeList,Subgraph[edgeList,nonTrivialComponentList]],*)
(*PlotLabel->"Non-trivial connected component"*)
(*]*)


(* ::Input:: *)
(*(*3D nonTrivialComponentListPlot*)*)
(*Clear@nonTrivialComponentListPlot3D*)
(*nonTrivialComponentListPlot3D=GraphPlot3D[Subgraph[edgeList,nonTrivialComponentList],PlotLabel->Style["Non-trivial connected component\n (What not to do)",16,Black]]*)


(* ::Subsubsection:: *)
(*Forest*)


(* ::Input:: *)
(*Graph@edgeList*)
(*GraphDisjointUnion@@ConnectedGraphComponents@edgeList*)


(* ::Subsubsection:: *)
(*Centrality*)


(* ::Text:: *)
(*ClosenessCentrality will give high centralities to vertices that are at a short average distance to every other reachable vertex.*)


(* ::Input:: *)
(*Clear@closenessPlot*)
(*MaximalBy[Transpose@{vertexList,ClosenessCentrality@edgeList},#[[2]]&,10][[;;,1]]*)
(*closenessPlot=GraphPlot[HighlightGraph[edgeList,%],PlotLabel->"Top 10 \"close\" vertices"]*)


(* ::Text:: *)
(*BetweennessCentrality will give high centralities to vertices that are on many shortest paths of other vertex pairs.*)


(* ::Input:: *)
(*MaximalBy[Transpose@{vertexList,BetweennessCentrality@edgeList},#[[2]]&,10][[;;,1]]*)
(*betweenPlot=GraphPlot[HighlightGraph[edgeList,%],PlotLabel->"Top 10 \"between\" vertices"]*)


(* ::Text:: *)
(*EigenvectorCentrality will give high centralities to vertices that are connected to many other well-connected vertices.*)


(* ::Input:: *)
(*MaximalBy[Transpose@{vertexList,EigenvectorCentrality@edgeList},#[[2]]&,10][[;;,1]]*)
(*GraphPlot[HighlightGraph[edgeList,%],PlotLabel->"Top 10 \"\" vertices"]*)


(* ::Subsection:: *)
(*Time Dependent*)


(* ::Subsubsection:: *)
(*Evolution*)


(* ::Input:: *)
(*Clear[networkEvolution,networkEvolutionGraphList]*)
(*networkEvolutionGraphList=FoldList[EdgeAdd[#,#2]&,Graph[vertexList,First@edgeList,GraphLayout->"CircularEmbedding"],edgeList[[2;;]]];*)


(* ::Input:: *)
(*networkEvolution=Transpose@{networkEvolutionGraphList,timeList};*)


(* ::Input:: *)
(*networkEvolutionPlotList=ParallelMap[*)
(*Rasterize[*)
(*GraphPlot[*)
(*#[[1]],*)
(*AspectRatio->1,*)
(*PlotLabel->#[[2]]*)
(*],*)
(*ImageSize->Large,*)
(*ImageResolution->144*)
(*]&,*)
(*networkEvolution*)
(*];*)


(* ::Subsubsection:: *)
(*Answer Time Series*)


(* ::Input:: *)
(*Clear@answerTimeSeriesPlot*)
(*answerTimeSeriesPlot=DateHistogram[*)
(*timeList,*)
(*"Hour",*)
(*PlotLabel->"Total Time Range: "<>ToString[DayCount@@MinMax@timeList]<>" Days",*)
(*Frame->True,*)
(*ChartStyle->RGBColor[0.24, 0.6, 0.33692049419863584`],*)
(*FrameLabel->"Interactions",*)
(*ImagePadding->{{40,20},{20,10}}*)
(*]*)


(* ::Section:: *)
(*Export*)


(* ::Subsection:: *)
(*Time Independent*)


(* ::Subsubsection:: *)
(*Communities*)


(* ::Input:: *)
(*Export["Network_Communities.png",Rasterize@communityPlot]*)


(* ::Subsubsection:: *)
(*Cycles*)


(* ::Input:: *)
(*Export["Network_Cycles.png",Rasterize@cyclePlot]*)


(* ::Subsubsection:: *)
(*Most Connected User Neighborhood*)


(* ::Input:: *)
(*Export["Network_MostConnectedUserNeighborhood.png",Rasterize@mostConnectedUserPlot]*)


(* ::Subsubsection:: *)
(*Connected Components*)


(* ::Input:: *)
(*Export["Network_ConnectedComponent.png",Rasterize@nonTrivialComponentListPlot]*)


(* ::Input:: *)
(*Export["Network_ConnectedComponent3D.png",Rasterize@nonTrivialComponentListPlot3D]*)


(* ::Subsection:: *)
(*Time Dependent*)


(* ::Subsubsection:: *)
(*Evolution*)


(* ::Input:: *)
(*Export["Network_Evolution.mp4",networkEvolutionPlotList,VideoEncoding->"HEVC-NVENC"(*,FrameRate->2*)]*)


(* ::Input:: *)
(*stream=VideoStream@"Network_Evolution.mp4";*)
(*Dynamic[stream["CurrentFrame"]]*)
(*VideoPlay[stream]*)


(* ::Input:: *)
(*RemoveVideoStream[]*)


(* ::Subsubsection:: *)
(*Answer Time Series*)


(* ::Input:: *)
(*Export["answerTimeSeriesPlot.png",Rasterize@answerTimeSeriesPlot]*)


(* ::Subsection:: *)
(*Archive Compression*)


(* ::Input:: *)
(*CreateArchive[FileNames["*.png"|"*.mp4"],"Network_Visualization.7z"]*)

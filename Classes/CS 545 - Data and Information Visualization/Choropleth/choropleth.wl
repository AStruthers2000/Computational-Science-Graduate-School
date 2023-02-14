(* ::Package:: *)

(* ::Title:: *)
(*CS 545: Data & Information Visualization*)


(* ::Subtitle:: *)
(*Assignment 3: Juxtaposed Choropleth Maps*)
(*Group 6: Nathan Chapman, Andrew Struthers, Kollin Trujillo, Gihanen, Nick Haviland*)
(*Date: Thu 9 Feb 2023*)


(* ::Section:: *)
(*Importing and Parsing Census Data*)


Clear[dataEncoded,itemCodeList,dataDecoded]
(*Importing data and parsing into dataset*)
dataEncoded=Import["https://www2.census.gov/programs-surveys/stc/datasets/2021/FY2021-Flat-File.txt",{"CSV","Dataset"},"HeaderLines"->{1,1}];
(*Import and parse item codes*)
itemCodeList=Rule@@@Import["https://www2.census.gov/programs-surveys/stc/technical-documentation/file-layout/taxitemcodesanddescriptions.xls","Data"][[1,2;;]];
(*Replace the item codes with their actual names*)
dataDecoded=Normal@Normal@dataEncoded/.Append[itemCodeList,"X"->"X"(*Missing value replacement*)]//Association//Dataset


(*Choose the 8 datasets with the fewest missing values, then randomly reorder them*)
Clear@dataCore
dataCore=SortBy[dataDecoded,Count["X"]][[;;8]]//RandomSample;
(*How many missing values are in the core data set*)"There are "<>ToString@Count[dataCore,"X",\[Infinity]]<>" missing data points"
(*What are the items in the core data set*)itemList=Normal@Keys@dataCore;


(* ::Section:: *)
(*Choropleth*)


(*Get state entities to be used in the choropleth*)
stateList=GeoEntities[Entity["Country", "UnitedStates"],"USState"];


Clear@choropleth
choropleth[item_String,Optional["UnitedStatesContinental"->q:(True|False),"UnitedStatesContinental"->False]]:=Rasterize[
GeoRegionValuePlot[
#->dataCore[item][#["StateAbbreviation"]]&/@If[q,DeleteCases[state_/;state["StateAbbreviation"]=="AK"||state["StateAbbreviation"]=="HI"],Identity]@stateList,
PlotLabel->item,
MissingStyle->Green,
TargetUnits->"US kDollars",
GeoLabels->(Tooltip[#1,"$"<>ToString@#4]&)
],
ImageResolution->300
]
choropleth[y_][x_]:=choropleth[x,y]


(*Test choropleth juxtaposition*)
choropleth["UnitedStatesContinental"->True]/@{"Motor Fuels Sales Tax","Motor Vehicles License"};
GraphicsRow[%,0,ImageSize->Full]


(* ::Section:: *)
(*Interactive Comparison*)


Manipulate[
GraphicsRow[
choropleth["UnitedStatesContinental"->cont]/@{item1,item2},
ImageSize->Full,
Spacings->0
],
{{cont,False,"Continental US"},{True,False}},
{{item1,itemList[[1]],"Left Dataset"},itemList[[;;4]]},
{{item2,itemList[[5]],"Right Dataset"},itemList[[5;;]]}
]

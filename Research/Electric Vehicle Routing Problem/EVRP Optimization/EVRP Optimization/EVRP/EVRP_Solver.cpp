#include "EVRP_Solver.h"
#include "GA\GAOptimizer.h"


EVRP_Solver::EVRP_Solver()
{
#if DEBUG
	capacity = 10;
	//nodes.push_back(Node{ 0, 0, 0 });
	nodes.push_back(Node{ 3, -5, 2 });
	nodes.push_back(Node{ 6, 4, 6 });
	nodes.push_back(Node{ -5, -1, 8 });
	nodes.push_back(Node{ 5, 3, 4 });
	nodes.push_back(Node{ -3, 4, 6 });
#else
	float nLocations, temp;

	std::ifstream file;
	char filename[STR_LEN] = ".\\EVRP\\Data_Sets\\EVRP TW\\c101C5.txt";
	bool isEVRP = true;

	file.open(filename);
	if (!file.is_open())
	{
		std::cout << "Failed to open data file, exiting" << std::endl;
		exit(1);
	}
	else
	{
		if (!isEVRP)
		{
			file >> nLocations >> provided_solution >> vehicleLoadCapacity;
			double x, y;
			file >> x >> y;
			//Node depot = Node{ x, y, 0 };
			//nodes.push_back(depot);
			for (int i = 0; i < nLocations; i++)
			{
				int demand;
				file >> temp >> x >> y >> demand;
				Node n = Node{ x, y, demand };
				customerNodes.push_back(n);
			}
		}
		else
		{
			std::string ID;
			char nodeType;
			std::string line;
			double x, y;
			int demand;
			int index = 0;
			data.customerStartIndex = -1;
			while (std::getline(file, line))
			{
				std::istringstream iss(line);
				if (!(iss >> ID >> nodeType >> x >> y >> demand))
				{
					char type = line[0];
					int pos = 0;
					std::string token;
					while ((pos = line.find('/')) != std::string::npos)
					{
						token = line.substr(0, pos);
						line.erase(0, pos + 1);
					}
					if (!token.empty())
					{
						float num = std::stof(token);
						switch (type)
						{
						case 'Q':
							vehicleBatteryCapacity = num;
							break;
						case 'C':
							vehicleLoadCapacity = num;
							break;
						case 'r':
							vehicleFuelConsumptionRate = num;
						default:
							break;
						}
					}
				}
				else
				{
					Node n = Node{ x, y, 0, false, index};
					switch (nodeType)
					{
					case 'f':
						n.demand = 0;
						n.isCharger = true;
						break;
					case 'c':
						if (data.customerStartIndex == -1) data.customerStartIndex = index;
						n.demand = demand;
						n.isCharger = false;
						break;
					default:
						//std::cout << "Undefined type: " << nodeType << std::endl;
						break;
					}
					nodes.push_back(n);
					index++;
				}
			}

			data = EVRP_Data{ nodes, vehicleBatteryCapacity, vehicleLoadCapacity, vehicleFuelConsumptionRate, data.customerStartIndex};
		}
	}
	file.close();
#endif

	int tot_demand = 0;
	for(const Node &node : nodes)
	{
		tot_demand += node.demand;
	}
	std::cout << "The minimum number of subtours is: " << std::ceil(double(tot_demand) / vehicleLoadCapacity) << std::endl;
}

EVRP_Solver::~EVRP_Solver()
{
}

std::vector<int> EVRP_Solver::SolveEVRP()
{
#if OLD_NOT_OPTIMIZED
	//initialize variables
	const int n = nodes.size();
	int remaining_capacity = capacity;
	int times_visited_depot = 0;

	vector<int> solution;
	vector<bool> visited(n, false);
	solution.reserve(n);

	//Start from the depot
	int current_node = 0;
	visited[current_node] = true;
	solution.push_back(current_node);

	//iterate until all nodes have been serviced
	while(!AllNodesVisited(visited))
	{
		//get the next closest servicable node
		int next_node = FindNearestServicableNode(visited, current_node, remaining_capacity);

		//No feasable next node can be found, so return to depot
		if (next_node == -1)
		{
#if VERBOSE
			cout << "Must head back to depot. Remaining capcity is: " << remaining_capacity << " and all nodes have higher demand" << endl << endl;
#endif
			solution.push_back(0);
			current_node = 0;
			remaining_capacity = capacity;
			times_visited_depot++;
		}
		//feasable node is found, visit node and remove demand from my capacity
		else
		{
#if VERBOSE
			cout << "Heading to node " << next_node << " from node " << current_node << ". My current capacity is: " << remaining_capacity << " and node " << next_node << " has demand " << nodes[next_node].demand << endl;
#endif
			visited[next_node] = true;
			solution.push_back(next_node);
			current_node = next_node;
			remaining_capacity -= nodes[next_node].demand;
		}
	}

	//always return back to the depot at the end of the route
	solution.push_back(0);

#if VERBOSE
	cout << "Heading back to depot because all nodes are satisfied" << endl;
	cout << "\n\n\n";
#endif

	return solution;
#else
	GAOptimizer* ga = new GAOptimizer();
	std::vector<int> optimalTour;
	int bestFitness;
	double bestDistance;
	ga->Optimize(data, optimalTour, bestFitness, bestDistance);
	std::cout << "There are " << bestFitness << " subtours in this route, with a total distance of this route is: " << bestDistance << std::endl;
	return optimalTour;

#endif
}

int EVRP_Solver::FindNearestServicableNode(std::vector<bool> visited, int current, int remaining_capacity) const
{
	int next = -1;
	double min_dist = std::numeric_limits<double>::max();

	//finds the closest node that we haven't visited that we still have capacity to serve
	for (int i = 0; i < visited.size(); i++)
	{
		if (!visited[i] && customerNodes[i].demand <= remaining_capacity)
		{
			double dist = Distance(customerNodes[current], customerNodes[i]);
			if (dist < min_dist)
			{
				min_dist = dist;
				next = i;
			}
		}
	}
	return next;
}

double EVRP_Solver::CalculateTotalDistance(const std::vector<int>& solution) const
{
	double tot = 0;
	for (int i = 1; i < solution.size(); i++)
	{
		tot += Distance(customerNodes[solution[i - 1]], customerNodes[solution[i]]);
	}
	return tot;
}

double EVRP_Solver::Distance(const Node& node1, const Node& node2) const
{
	return hypot(node1.x - node2.x, node1.y - node2.y);
}

bool EVRP_Solver::AllNodesVisited(std::vector<bool> visited) const
{
	int sum = std::accumulate(begin(visited), end(visited), 0);
	if (sum == visited.size()) return true;
	else return false;
}

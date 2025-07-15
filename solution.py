import informed.search as informed_search
import search as search
from itertools import combinations


class City:
    """Blueprint for creating City objects"""

    # Class constructor
    def __init__(self, name, cargo, has_truck=False):
        self.__name = name  # City name
        # Sort cargo objects in alphabetical order for better visual representation
        self.__cargo = sorted(cargo, key=lambda c: int(c.name[-1]))  # List of cargo at the city
        self.__has_truck = has_truck  # Bool --> whether current city has the truck

    # Getter for City name
    @property
    def name(self):
        return self.__name

    # Getter for list of cargo at the city
    @property
    def cargo(self):
        return self.__cargo

    # Getter for the 'has_truck' boolean, whether or not the truck is at the city
    @property
    def has_truck(self):
        return self.__has_truck

    # Method to remove cargo from the city
    def remove_cargo(self, cargo):
        return [c for c in self.__cargo if c not in cargo]

    # Method to add cargo to the city
    def add_cargo(self, cargo):
        return self.__cargo + cargo

    def describe(self):
        result = self.__name + ": "
        if self.__has_truck:
            result += 'TRUCK '
        if not self.__cargo:
            return result
        for cargo in self.__cargo:
            result += cargo.describe()
        return result


class Cargo:
    """Blueprint for creating Cargo objects"""

    # Class constructor
    def __init__(self, name, weight):
        self.__name = name  # Cargo name
        self.__weight = weight  # Cargo weight

    # Define a string to describe the cargo object
    def __repr__(self):
        return self.__name + " (" + str(self.__weight) + ") "

    # Getter for cargo name
    @property
    def name(self):
        return self.__name

    # Getter for cargo weight
    @property
    def weight(self):
        return self.__weight

    def describe(self):
        return self.__name + " (" + str(self.__weight) + ") "


class ProblemState(search.State):
    """
    Modelling the problem state
    Represents how the world looks like in the given state
    """

    # Class constructor
    def __init__(self, city_a, city_b, city_c):
        self.A = city_a  # 'A' city object
        self.B = city_b  # 'B' city object
        self.C = city_c  # 'C' city object

    # Helper method to find the city where the cargo is in the current given state
    def find_cargo(self, cargo):
        if cargo in self.A.cargo:
            return self.A.name
        elif cargo in self.B.cargo:
            return self.B.name
        elif cargo in self.C.cargo:
            return self.C.name
        else:
            return None

    # Calculate the hash value from the state's attributes that represent the given state (list of cargo for each city
    # and whether the city has the truck)
    def __hash__(self):
        return hash(
            (tuple(self.A.cargo), self.A.has_truck,
             tuple(self.B.cargo), self.B.has_truck,
             tuple(self.C.cargo), self.C.has_truck))

    # Method to return the String representation of the state object
    def __str__(self):
        result = ""
        for city in self.__dict__:
            result += self.__dict__[city].describe() + '\n'
        return result

    # Method to compare equality between two state objects using their hash values
    def __eq__(self, other):
        if not isinstance(other, ProblemState):
            # If the other object is not of class ProblemState, return False
            return False

        return self.__hash__() == other.__hash__()  # Return whether the two hash values are equal

    # Method to apply an action to the current state
    def apply_action(self, action):
        # Get the city objects from the current state - the city cargo is moved from, the city cargo is moved to and
        # the city that is left intact
        city_from = self.__dict__.get(action.city_from)
        city_to = self.__dict__.get(action.city_to)
        city_intact = self.__dict__.get([name for name in 'ABC' if name not in action.city_from+action.city_to][0])

        # Create the new city objects - sort them alphabetically by name for visual representation
        new_cities = sorted([
            City(city_from.name, city_from.remove_cargo(action.cargo)),
            City(city_to.name, city_to.add_cargo(action.cargo), True),
            City(city_intact.name, city_intact.cargo)
        ], key=lambda city: city.name)

        # Create the new problem state using the updated city objects
        return ProblemState(*new_cities)

    # Method to find all action-state pairs and return them as a list of ActionStatePair objects
    def successor(self):
        result = []
        # Find the city in the current state that has the truck
        city_with_truck = None
        for c in self.__dict__:
            if self.__dict__[c].has_truck:
                city_with_truck = self.__dict__[c]

        # Get all possible situations
        # Truck can go to either of the other two cities
        other_cities = [city for city in self.__dict__ if city != city_with_truck.name]

        # Get all possible cargo combinations within the allowed weight limit
        cargo_options = []
        current_cargos = city_with_truck.cargo

        for i in range(0, len(current_cargos) + 1):
            for j in combinations(current_cargos, i):
                # If the total weight exceeds the truck's load limit continue
                if sum([k.weight for k in j]) > ProblemAction.truck_load:
                    continue
                # Add valid cargo combination to the list
                cargo_options.append(j)

        # Combine the available cities to transfer to with possible cargo combinations to create ActionStatePairs
        for other_city in other_cities:
            for cargo_option in cargo_options:
                action = ProblemAction(city_with_truck.name, other_city, list(cargo_option))  # Create the action
                next_state = self.apply_action(action)  # Apply the action to the current state
                # Create a new ActionStatePair using the action and the resulting state object
                action_state_pair = search.ActionStatePair(action, next_state)
                # Add the ActionStatePair to the result list
                result.append(action_state_pair)

        return result  # Return the result list


class ProblemAction(search.Action):
    """Modelling actions that can be apply to a state"""

    # Define action costs for the basic problem
    """fixed_action_cost = {
        'AB': 80.0,
        'BA': 80.0,
        'BC': 20.0,
        'CB': 20.0,
        'AC': 50.0,
        'CA': 50.0
    }

    variable_action_cost = {
        'AB': 1.0,
        'BA': 1.0,
        'BC': 4.0,
        'CB': 4.0,
        'AC': 2.0,
        'CA': 2.0
    }"""

    # Define action costs for the general problem (1)
    """fixed_action_cost = {
        'AB': 1.0,
        'BA': 1.0,
        'BC': 2.0,
        'CB': 2.0,
        'AC': 3.0,
        'CA': 3.0
    }

    variable_action_cost = {
        'AB': 2.0,
        'BA': 2.0,
        'BC': 0.8,
        'CB': 0.8,
        'AC': 0.3,
        'CA': 0.3
    }"""

    # Define action costs for the general problem (2)
    fixed_action_cost = {
        'AB': 2.0,
        'BA': 2.0,
        'BC': 2.0,
        'CB': 2.0,
        'AC': 80.0,
        'CA': 80.0
    }

    variable_action_cost = {
        'AB': 1.0,
        'BA': 1.0,
        'BC': 4.0,
        'CB': 4.0,
        'AC': 2.0,
        'CA': 2.0
    }

    # truck_load = 15.0  # Truck load limit for the basic problem
    # truck_load = 50.0  # Truck load limit for the general problem (1)
    truck_load = 30.0  # Truck load limit for general problem (2)

    # Class constructor
    def __init__(self, city_from, city_to, cargo):
        super().__init__()
        self.__city_from = city_from  # City where the cargo is moved from
        self.__city_to = city_to  # City where the cargo is moved to
        self.__cargo = cargo  # The list of cargo to be moved
        self.__total_load = self._calculate_total_load()  # The total load of all cargo
        self.__total_cost = self._calculate_total_cost()  # The total cost incurred by the action

    # Define a String representation of the action object
    def __str__(self):
        cargos = " ".join([c.describe() for c in self.__cargo])
        return f'Move CITY_{self.city_from} -> CITY_{self.__city_to}: {cargos} Load: {self.__total_load} Cost: ' \
               f'{self.__total_cost}'

    # Getter for the city the cargo is moved from
    @property
    def city_from(self):
        return self.__city_from

    # Getter for the city the cargo is moved to
    @property
    def city_to(self):
        return self.__city_to

    # Getter for the cargo
    @property
    def cargo(self):
        return self.__cargo

    # Method to calculate the total cost incurred by the action
    def _calculate_total_cost(self):
        # Get the corresponding fixed cost from the lookup dictionary
        fixed_cost = self.fixed_action_cost[self.city_from + self.city_to]
        # Get the corresponding variable cost from the lookup dictionary
        variable_cost = self.variable_action_cost[self.city_from + self.city_to]
        # Calculate the total cost of action
        self.cost = fixed_cost + variable_cost * self.__total_load
        return self.cost  # Return the total cost

    # Method to calculate the total load for all cargo moved as part of the action
    def _calculate_total_load(self):
        return sum([cargo.weight for cargo in self.__cargo])


class TransportBestFirstSearchProblem(informed_search.BestFirstSearchProblem):

    # Method to check whether the current state is the goal state
    def isGoal(self, state):
        return self.goalState == state

    # Evaluation function f(n) to get the estimated cost of the cheapest path from start to goal through a given node
    def evaluation(self, node):
        return node.getCost() + self.heuristic(node)  # Return g(n) - known cost + h(n) - estimated remaining cost
        # return self.heuristic(node)  # Greedy best-first search using heuristic h(n) as the evaluation function
        # return node.getCost()  # Breadth-first search using g(n) as the evaluation function

    # Heuristic function h(n) to estimate the remaining cost from given node to goal state
    def heuristic(self, node):
        h1_cost = 0  # h1(n) total
        h2_cost = 0  # h2(n) total
        # Iterate through each city in the state of the current node
        for city in node.state.__dict__:
            destinations = set()
            for cargo in node.state.__dict__[city].cargo:
                # For each cargo in the current city find the destination city it resides in the goal state
                destination = self.goalState.find_cargo(cargo)
                # If the cargo is not in it's goal state destination
                if destination != city:
                    # Add the destination to the destinations set
                    destinations.add(destination)
                    # Add the variable cost of moving the cargo to the goal state destination
                    h1_cost += ProblemAction.variable_action_cost[city + destination] * cargo.weight
                    h2_cost += ProblemAction.variable_action_cost[city + destination] * cargo.weight
            # Add fixed cost to move cargos to their destination city
            for dest in destinations:
                h2_cost += ProblemAction.fixed_action_cost[city + dest]

        return h1_cost  # Return the estimated cost for using only 'h1' heuristic
        # return max(h1_cost, h2_cost)  # Return the more informed heuristics with higher cost


if __name__ == '__main__':

    # BASIC PROBLEM CONFIGURATION
    """c1 = Cargo('c1', 2.5)
    c2 = Cargo('c2', 7.5)
    c3 = Cargo('c3', 3.0)
    c4 = Cargo('c4', 8.5)
    c5 = Cargo('c5', 10.0)
    c6 = Cargo('c6', 12.0)

    initial_state = ProblemState(
        City('A', [c1, c2, c3], True),
        City('B', [c4, c5]),
        City('C', [c6])
    )

    test_state = ProblemState(
        City('A', [c1, c2, c3], True),
        City('B', [c4, c5]),
        City('C', [c6])
    )

    goal_state = ProblemState(
        City('A', [c4, c6], True),
        City('B', [c1, c2]),
        City('C', [c3, c5])
    )"""

    # GENERAL (1) PROBLEM CONFIGURATION
    """c1 = Cargo('c1', 1.5)
    c2 = Cargo('c2', 3.5)
    c3 = Cargo('c3', 5.5)
    c4 = Cargo('c4', 10.0)

    initial_state = ProblemState(
        City('A', [c1, c2], True),
        City('B', [c3, c4]),
        City('C', [])
    )

    goal_state = ProblemState(
        City('A', []),
        City('B', []),
        City('C', [c1, c2, c3, c4], True)
    )"""

    # GENERAL (2) PROBLEM CONFIGURATION
    c1 = Cargo('c1', 1.5)
    c2 = Cargo('c2', 3.5)
    c3 = Cargo('c3', 5.5)
    c4 = Cargo('c4', 10.0)

    initial_state = ProblemState(
            City('A', [c1, c2, c3, c4], True),
            City('B', []),
            City('C', [])
        )

    goal_state = ProblemState(
            City('A', []),
            City('B', []),
            City('C', [c1, c2, c3, c4], True)
        )


    # Instantiate the search problem
    problem = TransportBestFirstSearchProblem(initial_state, goal_state)
    # Carry out the search
    path = problem.search()

    if not path:
        print('No solution')
    else:
        # Output the path, the number of nodes visited and the total cost for the path
        print(path)
        print(f'Nodes visited: {problem.nodeVisited}')
        print(f'Cost: {path.cost}')

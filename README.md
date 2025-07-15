# CMM3038 - Artificial Intelligence for Problem Solving

***This project implements the A* algorithm by the design and development of an intelligent game-playing system to solve a puzzle***

### The Problem Domain - 'Transport Puzzle'
A transport company serves clients by moving cargos between 3 cities:
  - There are 2 cities : A, B, and C
  - Each cargo has a weight (e.g.: 2 tonnes)
  - A cargo starts in a city but needs to be transported to a destination city
  - The company has one truck to move cargos with a maximum load (e.g.: 15 tonnes) - the total weights of the cargos cannot exceed the maximum load
  - To move cargos between 2 cities:
      * There is a fixed cost which depends on the toll, and the minimum fuel cost required to cover the distance
      * There is also a variable cost which depends on the total weight of the cargos to transport
      * The costs only depend on the city pair, not the travelling direction
   
### Aim:
Given the initial location of the truck and cargos, derive a plan to move the cargos to their destination with the lowest total cost

<img width="2100" height="678" alt="image" src="https://github.com/user-attachments/assets/addb45bf-7960-4d11-bb94-b48cd2e84987" />

_The code solution for the transport puzzle can be found in the 'solution.py' file, while 'CM3038 Coursework (1608118)' provides a detailed explanation of the solution developed_

import pprint
from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    NONE = 0
    DIAG = 1
    UP = 2
    LEFT = 3


@dataclass
class alignment_slot:
    cost: int
    i: int = 0
    j: int = 0
    determined: 'alignment_slot' = None
    direction: Direction = Direction.NONE
    
x_in = input("Enter string 1: ")
y_in = input("Enter string 2: ")

x = "AACAGTTACC" if len(x_in) == 0 else x_in
y = "TAAGGTCA" if len(y_in) == 0 else y_in

m = len(x)
n = len(y)

alignment_table = [[alignment_slot(0) for _ in range(n + 1)] for _ in range(m + 1)]
memo = {}

def opt(i, j) -> int:
    if (i, j) in memo:
        return memo[(i, j)]

    alignment_table[i][j].i = i
    alignment_table[i][j].j = j
    
    if i == m:
        alignment_table[i][j].cost = 2*(n-j)
        return 2*(n-j)
    elif j == n:
        alignment_table[i][j].cost = 2*(m-i)
        return 2*(m-i)
    else:
        penalty = 0 if x[i] == y[j] else 1

    diag = opt(i + 1, j + 1) + penalty
    inc_i = opt(i + 1, j) + 2
    inc_j = opt(i, j + 1) + 2
    result = min(diag, inc_i, inc_j)
    alignment_table[i][j].cost = result

    if diag < min(inc_i, inc_j):
        alignment_table[i][j].determined = alignment_table[i + 1][j + 1]
        alignment_table[i][j].direction = Direction.DIAG
        
    elif inc_i < inc_j:
        alignment_table[i][j].determined = alignment_table[i + 1][j]
        alignment_table[i][j].direction = Direction.UP
        
    else:
        alignment_table[i][j].determined = alignment_table[i][j + 1]
        alignment_table[i][j].direction = Direction.LEFT
    
    memo[(i, j)] = result
    return result


result = opt(0, 0)

slot = alignment_table[0][0]
slots = []
while slot.determined != None:
    slots.append(slot)
    slot = slot.determined
slots.append(slot)
slots.reverse()

x_p = ["" for _ in range(len(slots)-1)]
y_p = ["" for _ in range(len(slots)-1)]
index = len(slots)-2

for i in range(len(slots)-1):
    next_slot = slots[i+1]
    
    if next_slot.direction == Direction.DIAG:
        x_p[index] = x[next_slot.i]
        y_p[index] = y[next_slot.j]

    elif next_slot.direction == Direction.UP:
        x_p[index] = x[next_slot.i]
        y_p[index] = "-"

    else:
        x_p[index] = "-"
        y_p[index] = y[next_slot.j]

    index -= 1

print(f"\nOptimal alignment: \n\n{''.join(x_p)}")
print(f"{''.join(y_p)}")
print(f"\nCost of alignment: {result}")

---
title: 1518-Water Bottles
date: 2022-08-14
permalink: /posts/leetcode/1518/
read_time: false
show_in_archive: false
tags: 
    leetcode
---

## Optimized Bottle Exchange
Even though this is an easy solution  i absoluetly loved this elegant solution that i thoough of. 
### Approach

In this optimized approach, we use math to gulp down as many bottles as possible before exchanging them for full ones. By maximizing each exchange, we ensure efficiency.

### Steps

1. Start with a count of empty bottles consumed set to zero.
2. Continue until the number of full bottles is greater than the exchange rate:
   - Calculate the maximum number of full bottles we can gulp down in one go.
   - Add these to our consumed count.
   - Subtract the corresponding empty bottles from the total.
   - Exchange the empty bottles for new full ones.
3. Return the total number of bottles consumed, including any remaining full bottles.

### Implementation

```python
def num_water_bottles(numBottles: int, numExchange: int) -> int:
    consumedBottles = 0
    while numBottles >= numExchange:
        K = numBottles // numExchange
        consumedBottles += numExchange * K
        numBottles = numBottles - numExchange * K + K
    return consumedBottles + numBottles
```
### Complexity

	- Time: Efficient, logarithmic relative to the number of bottles.
	- Space: Minimal, only requires a few variables.

### Question Pattern Resembles 
	 - Greedy Approach:  You could say it is greedy but i wouldnt classify it as such overall i cant pinpoint it to any particular pattern. 

That reminds me Enjoy drinking responsibly and efficiently!
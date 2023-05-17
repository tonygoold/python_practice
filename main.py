from collections import deque
from functools import reduce
from heapq import *
from binary_heap import BinaryHeap
from binary_tree import BinaryNode
from binary_search import binary_search
from interval import Interval
from linked_list import Link, LinkedList
from lru_cache import LRUCache

def kth_largest(k, ns):
  """Find the kth largest element in ns."""
  if len(ns) < k:
    return None
  heap, tail = ns[:k], ns[k:]
  heapify(heap)
  for n in tail:
    heappushpop(heap, n)
  return heappop(heap)

def kth_largest_custom(k, ns):
  """Find the kth largest element in ns."""
  heap = BinaryHeap(lambda x, y: x <= y)
  if len(ns) < k:
    return None
  for n in ns[:k]:
    heap.push(n)
  for n in ns[k:]:
    heap.push(n)
    heap.pop()
  return heap.pop()

def k_closest_1(ns, k, x):
  """Find the k elements in ns closest to x, where ns is sorted.
  Return the elements in sorted order.
  
  We can track this as the index of the start of the subarray and the sum of
  its absolute differences from x. Each time we advance the index, we subtract
  the first absolute difference and add the next absolute difference."""
  last = len(ns) - k
  if last < 0:
    return None
  mindiff = sum([abs(n - x) for n in ns[:k]])
  i = 0
  while i < last:
    nextdiff = mindiff + abs(ns[i + k] - x) - abs(ns[i] - x)
    if nextdiff > mindiff:
      break
    mindiff = nextdiff
    i += 1
  return ns[i:i+k]

def k_closest_2(ns, k, x):
  """Find the k elements in ns closest to x, where ns is sorted.
  Return the elements in sorted order.
  
  First, we find the index of the element in ns closest to x. We extend this
  into a range by checking whether to add the element before or after the
  range, based on which has the smallest difference from x, until we have k
  elements."""
  i, n = binary_search(ns, x, nearest = True)
  if i is None:
    return None
  l = r = i
  while r + 1 - l < k:
    # If either bound is at the end, we can only grow in one direction and can
    # immediately return a solution.
    if l == 0:
      return ns[:k]
    elif r + 1 == len(ns):
      return ns[-k:]
    elif abs(x - ns[l - 1]) <= abs(x - ns[r + 1]):
      l -= 1
    else:
      r += 1
  return ns[l:r+1]

def find_sum_of_two(ns, x):
  """Find any two elements in ns that sum to x and return them.
  The elements must have distinct indices, so x/2 can only be in the solution
  if it occurs at least twice in the input array."""
  seen = set()
  for n in ns:
    if (x - n) in seen:
      return x - n, n
    seen.add(n)
  return None, None

# Question 1: Find the kth largest element.

in_order = list(range(1, 101))
# reverse_order = [101 - n for n in range(1, 101)]

largest = kth_largest(23, in_order)
if largest == 78:
  print("Correct! The 23th largest element in [1, 100] is 78")
else:
  print("Incorrect!", largest, "!= 78")

# largest = kth_largest_custom(23, in_order)
# if largest == 78:
#   print("Correct! The 23th largest element in [1, 100] is 78")
# else:
#   print("Incorrect!", largest, "!= 78")

# largest = kth_largest(23, reverse_order)
# if largest == 78:
#   print("Correct! The 23th largest element in [1, 100] is 78")
# else:
#   print("Incorrect!", largest, "!= 78")

# largest = kth_largest_custom(23, reverse_order)
# if largest == 78:
#   print("Correct! The 23th largest element in [1, 100] is 78")
# else:
#   print("Incorrect!", largest, "!= 78")

# Question 2: Find the subarray of length k whose elements are closest to n.

closest_4_3 = k_closest_1([1,2,3,4,5], 4, 3)
if sum([abs( 3 - x) for x in closest_4_3]) == 4:
  print("Correct! The closest 4-array to 3 is", closest_4_3)
else:
  print("Incorrect! The closest 4-array to 3 is not", closest_4_3)

# closest_3_6 = k_closest_1([2,4,5,6,9], 3, 6)
# print(closest_3_6)

# closest_4_3 = k_closest_2([1,2,3,4,5], 4, 3)
# print(closest_4_3)

closest_3_6 = k_closest_2([2,4,5,6,9], 3, 6)
if sum([abs(6 - x) for x in closest_3_6]) == 3:
  print("Correct! The closest 3-array to 6 is", closest_3_6)
else:
  print("Incorrect! The closest 3-array to 6 is not", closest_3_6)

# Question 3: Find a pair of values in a list, which sum to n.

pair1, pair2 = find_sum_of_two([2, 7, 11, 15], 9)
if pair1 is None or pair2 is None:
  print("Incorrect! No pair found")
else:
  print("Corect!", pair1, "+", pair2, "= 9")

# Question 4: Delete an element from a linked list.

ll = LinkedList()
ll.insert_head(9)
ll.insert_head(1)
ll.insert_head(5)
ll.insert_head(4)
if ll.size() != 4:
  print("Incorrect! Size calculation is wrong.")
if not ll.contains(5):
  print("Incorrect! List does not contain target value.")
ll.delete(5)
if ll.contains(5):
  print("Incorrect! List still contains target value.")
elif ll.size() != 3:
  print("Incorrect! List did not shrink after removal.")
else:
  print("Correct! List removed target value.")

# Question 5: Copy linked list with "arbitrary" pointer.

def deep_copy_arbitrary_pointer(head):
  # Short circuit to avoid conditional inside the loop
  if head is None:
    return None
  # Run a first pass, copying the list as is and creating a mapping
  # from old pointers to new pointers
  mapping = dict()
  node = head
  newhead = newnode = Link(node.key, None)
  mapping[node] = newnode
  while node.next is not None:
    node = node.next
    newnode.next = Link(node.key, None)
    newnode = newnode.next
    mapping[node] = newnode
  # Second pass to update arbitrary pointer
  node = newhead
  while node is not None:
    key, ptr = node.key
    if ptr is not None:
      ptr = mapping[ptr]
      node.key = key, ptr
    node = node.next
  return newhead

def sum_arbitrary_pointer(head):
  # Helper method to check validity
  sum = 0
  while head is not None:
    key, _ = head.key
    sum += key
    head = head.next
  return sum

# Set the key to a pair (value, arbitrary_pointer)
n0 = Link((7, None), None)
n1 = Link((13, n0), None)
n4 = Link((1, n0), None)
n2 = Link((11, n4), None)
n3 = Link((10, n2), None)
n3.next = n4
n2.next = n3
n1.next = n2
n0.next = n1

expected_sum = sum_arbitrary_pointer(n0)
actual_sum = sum_arbitrary_pointer(deep_copy_arbitrary_pointer(n0))

n0.key = 0, None
n1.key = 0, None
n2.key = 0, None
n3.key = 0, None
n4.key = 0, None

if sum_arbitrary_pointer(n0) != 0:
  print("Incorrect! Failed to zero out original list.")
elif expected_sum == 0:
  print("Incorrect! Expected sum should be non-zero.")
elif actual_sum != expected_sum:
  print("Incorrect!", actual_sum, "is not expected value", expected_sum)
else:
  print("Correct! Arbitrary pointer list is a deep copy.")

# Question 6: Mirror a binary tree
# Turn:
#         40
#       /    \
#    100      400
#   /   \    /   \
# 150   50  300  600
#
# Into:
#         40
#       /    \
#    400      100
#   /   \    /   \
# 600  300  50   150

bintree = BinaryNode(40,
  BinaryNode(100,
    BinaryNode(150),
    BinaryNode(50)
  ), BinaryNode(400,
    BinaryNode(300),
    BinaryNode(600)
  )
)

orig = []
bintree.postorder(lambda node: orig.append(node.key))
if orig != [150, 50, 100, 300, 600, 400, 40]:
  print("Incorrect! Post-order traversal is not correct.")
def swap_children(node):
  node.left, node.right = node.right, node.left
bintree.postorder(swap_children)
swapped = []
bintree.postorder(lambda node: swapped.append(node.key))
if swapped == [600, 300, 400, 50, 150, 100, 40]:
  print("Correct! Binary tree was mirrored.")
else:
  print("Incorrect! Binary tree was not mirrored.")

# Question 7: Check if two binary trees are identical.

def binary_trees_equal_recursive(t1, t2):
  if t1 is None or t2 is None:
    return t1 is None and t2 is None
  elif t1.key != t2.key:
    return False
  return binary_trees_equal_recursive(t1.left, t2.left) and \
    binary_trees_equal_recursive(t1.right, t2.right)

def binary_trees_equal_nonrecursive(t1, t2):
  # Perform a breadth-first comparison using deques
  q1 = deque()
  q1.append(t1)
  q2 = deque()
  q2.append(t2)
  # Assuming len(q1) is O(n), instead stop when popleft() throws
  while True:
    try:
      n1 = q1.popleft()
      n2 = q2.popleft()
      if n1 is None or n2 is None:
        if n1 is not None or n2 is not None:
          return False
      elif n1.key != n2.key:
        return False
      else:
        q1.append(n1.left)
        q1.append(n1.right)
        q2.append(n2.left)
        q2.append(n2.right)
    except:
      return True

bintree1 = BinaryNode(1, BinaryNode(2), BinaryNode(3))
bintree2 = BinaryNode(1, BinaryNode(2), BinaryNode(3))
# Both functions should return true
trees_equal = binary_trees_equal_recursive(bintree1, bintree2) and \
  binary_trees_equal_nonrecursive(bintree1, bintree2)
if trees_equal:
  print("Correct! The first pair of binary trees are equal.")
else:
  print("Incorrect! The first pair of binary trees should not be equal.")

bintree1 = BinaryNode(1, BinaryNode(2), BinaryNode(1))
bintree2 = BinaryNode(1, BinaryNode(1), BinaryNode(2))
# Neither function should return true
trees_equal = binary_trees_equal_recursive(bintree1, bintree2) or \
  binary_trees_equal_nonrecursive(bintree1, bintree2)
if trees_equal:
  print("Inorrect! The second pair of binary trees should not be equal.")
else:
  print("Correct! The second pair of binary trees are not equal.")

# Question 8: String segmentation

def can_segment(input, words, offset = 0, checked = None):
  if checked is None:
    checked = {len(input): True}
  # Can the input be split into words contained in the 'words' set?
  # We can solve this recursively by observing that input can be segmented if,
  # for some k, input[:k] is a word and input[k:] can be segmented. By
  # memoizing answers, we can avoid re-checking input[k:]. This only helps
  # when there is some prefix input[:k] that can be segmented more than one
  # way, otherwise input[k:] would only be checked once anyway.

  # Base case: Already solved (includes offset == len(input)).
  if offset in checked:
    return checked[offset]
  for word in words:
    if not input.startswith(word, offset):
      continue
    valid = checked[offset] = can_segment(input, words, offset + len(word), checked)
    if valid:
      return True
  return False

if can_segment("applepenapple", set(["apple", "pen"])):
  print("Correct! Can be segmented.")
else:
  print("Incorrect! Could have been segmented.")

# Question 9: Find all palindrome substrings.
# Do not include trivial, one-letter palindromes.

def count_palindromes(input):
  n = len(input)
  # Initialize matrices with base case values. When i == j, we consider it to
  # be a palindrome (is_palindrome), but we don't count it (counted).
  counted = [[0 for _ in range(n)] for _ in range(n)]
  is_palindrome = [[True if x == y else False for x in range(n)] for y in range(n)]

  # Count all palindromes of length 2
  for i in range(n-1):
    if input[i] == input[i+1]:
      counted[i][i+1] = 1
      is_palindrome[i][i+1] = True

  # Now repeat again for length l 3..n
  for l in range(3, n+1):
    # All the substrings of length l from [i, l) to [n-l, i+l)
    for i in range(n - l + 1):
      j = i + l - 1
      # input[i:j+1] has at least as many palindromes as its substrings,
      # which is equal to the number of palindromes by excluding either end
      # minus the number of palindromes by excluding both ends (to avoid
      # counting them twice).
      count = counted[i][j-1] + counted[i+1][j] - counted[i+1][j-1]
      # If the ends match and input[i+1:j] is a palindrome, then the substring
      # input[i:j+1] is a palindrome.
      if input[i] == input[j] and is_palindrome[i+1][j-1]:
        is_palindrome[i][j] = True
        count += 1
      counted[i][j] = count

  return counted[0][n-1]

if count_palindromes("abaab") == 3:
  print("Correct! There are 3 non-trivial palindromes.")
else:
  print("Incorrect! There should be 3 non-trivial palindromes")

# Question 10: Largest sum subarray

# Note: This problem has a well-known solution called Kadane's algorithm.

# Thoughts:
# We can exclude any end-values that are negative.
# Assumption: The solution can be empty if the input is empty or the input
# contains only negative values.
# Assumption: Leading and trailing zeroes can be included.
#
# At a given index j, there are three possibilities:
# 1. The solution includes the largest sum subarray up to j-1 and includes j.
# 2. The solution includes the largest sum subarray up to j-1 and excludes j.
# 3. The solution does not include the largest sum subarray up to j-1.
#
# The first two possibilities can be distinguished by whether a k > j exists
# such that the largest sum subarray also includes k. The third possibility
# can be distinguished if the sum up to j is negative, in which case either
# the previous solution was the largest or the solution does not include the
# previous solution. This matches Kadane's solution, which does not track the
# subarray itself, only the largest sum.

def largest_sum_subarray(ns):
  cur_i = best_i = 0
  best_j = -1
  cur_sum = 0
  best_sum = 0
  for j, nj in enumerate(ns):
    next_sum = cur_sum + nj
    if next_sum < 0:
      # I think this is right... we want to skip negative values, which
      # we must have if the sum is less than zero.
      cur_i = j + 1
    cur_sum = max(0, next_sum)
    if cur_sum > best_sum:
      best_i = cur_i
      best_j = j
      best_sum = cur_sum
  # This should work even if a solution isn't found
  return ns[best_i:best_j+1]

largest = largest_sum_subarray([-2,1,-3,4,-1,2,1,-5,4])
if largest == [4,-1,2,1]:
  print("Correct! The largest sum subarray is", largest, "with sum", sum(largest))
else:
  print("Incorrect!", largest, "is not the largest sum subarray")

largest = largest_sum_subarray([1, 2, 3, -7, 8])
if largest == [8]:
  print("Correct! The largest sum subarray is", largest, "with sum", sum(largest))
else:
  print("Incorrect!", largest, "is not the largest sum subarray")

# Question 11: Determine if the input string is a valid number.

def valid_number(s):
  # Bonus: Don't just validate, also parse. Return False if invalid.
  n = 0
  sign = 1
  decimal = None
  for i, c in enumerate(s):
    match c:
      case '-':
        if i != 0:
          return False
        sign = -1
      case '+':
        if i != 0:
          return False
      case '.':
        if decimal is not None:
          return False
        decimal = 0.1
      case '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9':
        d = ord(c) - ord('0')
        if decimal is None:
          n = 10 * n + d
        else:
          n += decimal * d
          decimal *= 0.1
      case _:
        return False
  return sign * n

if valid_number("4.325") > 4.324: # Allow for precision errors
  print("Correct! 4.325 is a valid number.")
else:
  print("Incorrect! 4.325 should have been a valid number.")

if valid_number("1.1.1") is False:
  print("Correct! 1.1.1 is not a valid number.")
else:
  print("Incorrect! 1.1.1 should not have been a valid number.")

if valid_number("-7") == -7:
  print("Correct! -7 is a valid number.")
else:
  print("Incorrect! -7 should have been a valid number.")

# Question 12: Print balanced brace combinations.

# Simple observation: At each step, there are two options:
# 1. If there is at least one unbalanced open brace, close one.
# 2. If there are remaining braces to open, open one.
# Recursively do one or both until nothing remains.

def balanced_braces(remaining, open = 0, prefix = ""):
  # Base case: Done
  if remaining == 0 and open == 0:
    return [prefix]
  result = []
  if remaining > 0:
    result.extend(balanced_braces(remaining - 1, open + 1, prefix + "("))
  if open > 0:
    result.extend(balanced_braces(remaining, open - 1, prefix + ")"))
  return result

braces = balanced_braces(3)
if len(braces) == 5:
  print("Correct! There are 5 ways to balance 3 pairs of braces:", braces)
else:
  print("Incorrect! There should be 5 ways to balance 3 pairs of braces.")

# Question 13: Implement an LRU cache.

cache = LRUCache(2)
cache.put(1, 1) # [1]
cache.put(2, 2) # [1, 2]
if cache.get(1) != 1: # [2, 1]
  print("Incorrect! Cache should have returned 1.")
cache.put(3, 3) # [1, 3]
if cache.get(2) is not None:
  print("Incorrect! Cache should have evicted 2.")
cache.put(4, 4) # [3, 4]
if cache.get(1) is not None:
  print("Incorrect! Cache should have evicted 1.")
if cache.get(3) == 3 and cache.get(4) == 4: # [4, 3] -> [3, 4]
  print("Correct! Cache returned 3 and 4.")
else:
  print("Incorrect! Cache should have returned 3 and 4.")

# Question 14: Find low/high index of target in sorted list.

def find_range_linear(ns, x):
  iter = enumerate(ns)
  for i, n in iter:
    if n == x:
      for j, n in iter:
        if n != x:
          return i, j-1
      return i, len(ns) - 1
  return -1, -1

def find_range_binary(ns, x):
  # Use binary search to find the two ends, starting with the lower end
  start, end = 0, len(ns) - 1
  # Empty array check
  if end < 0:
    return -1, -1
  # This is like a regular binary search, except we're searching for a value
  # infinitessimally smaller than x, so we treat an exact match as too high.
  while start <= end:
    mid = (start + end) // 2
    if x > ns[mid]:
      start = mid + 1
    else: # x <= mid
      end = mid - 1
  # End condition: start >= x, end < x. Therefore, the value was found if and
  # only if start is a valid index and ns[start] equals the value.
  if start < len(ns) and ns[start] != x:
    return -1, -1
  low = start
  # Same as above, but now we want infinitessimally larger than x to find the
  # upper bound.
  start, end = 0, len(ns) - 1
  while start <= end:
    mid = (start + end) // 2
    if x >= ns[mid]:
      start = mid + 1
    else: # x < mid
      end = mid - 1
    # End condition: start > x, end <= x. We know a value exists at this
    # point, so we can skip checking ns[end].
  return low, end

# Linear search: O(n)

low, high = find_range_linear([5, 7, 7, 8, 8, 10], 8)
if low == 3 and high == 4:
  print("Correct! The range is [3, 4].")
else:
  print("Incorrect! The range should have been [3, 4], got [%d, %d]." % (low, high))

low, high = find_range_linear([5, 7, 7, 8, 8, 10], 10)
if low == 5 and high == 5:
  print("Correct! The range is [5, 5].")
else:
  print("Incorrect! The range should have been [5, 5], got [%d, %d]." % (low, high))

low, high = find_range_linear([5, 7, 7, 8, 8, 10], 6)
if low == -1 and high == -1:
  print("Correct! The value was not found.")
else:
  print("Incorrect! The value should not have been found, got [%d, %d]." % (low, high))

# Binary search: O(lg n)

low, high = find_range_binary([5, 7, 7, 8, 8, 10], 8)
if low == 3 and high == 4:
  print("Correct! The range is [3, 4].")
else:
  print("Incorrect! The range should have been [3, 4], got [%d, %d]." % (low, high))

low, high = find_range_binary([5, 7, 7, 8, 8, 10], 10)
if low == 5 and high == 5:
  print("Correct! The range is [5, 5].")
else:
  print("Incorrect! The range should have been [5, 5], got [%d, %d]." % (low, high))

low, high = find_range_binary([5, 7, 7, 8, 8, 10], 6)
if low == -1 and high == -1:
  print("Correct! The value was not found.")
else:
  print("Incorrect! The value should not have been found, got [%d, %d]." % (low, high))

# Question 15: Merge overlapping intervals.

def merge_intervals(its: list[Interval]):
  # I think the question assumes this. If not, I believe this is still the
  # most efficient way to merge.
  its.sort()
  def merge(result: list[Interval], x: Interval):
    if len(result) == 0:
      return [x]
    last = result[-1]
    if (merged := last.merge(x)) is not None:
      result.pop()
      result.append(merged)
    else:
      result.append(x)
    return result
  return reduce(merge, its, [])

input = [
  Interval(1, 3),
  Interval(2, 6),
  Interval(8, 10),
  Interval(15, 18),
]
merged = merge_intervals(input)
if len(merged) == 3 and merged[0] == Interval(1, 6):
  print("Correct! Intervals were merged correctly.")
else:
  print("Incorrect! The intervals should not be %s." % merged)

input = [
  Interval(1, 4),
  Interval(4, 5),
]
merged = merge_intervals(input)
if len(merged) == 1 and merged[0] == Interval(1, 5):
  print("Correct! Intervals were merged correctly.")
else:
  print("Incorrect! The intervals should not be %s." % merged)

# Question 16: Path sum.

# Assumption: Values are non-negative, to avoid recursing into nodes
# when the sum would exceed the target value. If this assumption is
# incorrect, then only exact equality is a stopping condition.
def has_path(root: BinaryNode, x: int):
  # This check can be eliminated by wrapping the function in one that
  # checks for the trivial case of searching for 0.
  if x == 0:
    return True

  x -= root.key
  if x == 0:
    return True
  elif x < 0:
    return False
  return (root.left is not None and has_path(root.left, x)) or \
    root.right is not None and has_path(root.right, x)

tree = BinaryNode(5,
  BinaryNode(4,
    BinaryNode(11,
      BinaryNode(7),
      BinaryNode(2)
    ),
    None
  ),
  BinaryNode(8,
    BinaryNode(13),
    BinaryNode(4,
      None,
      BinaryNode(1)
    )
  )
)

if has_path(tree, 22):
  print("Correct! The tree has a path of sum 22.")
else:
  print("Incorrect! The tree should have a path of sum 22.")

# Alter the tree so the solution 5+4+11+2=22 becomes 5+4+11+3=23
tree.left.left.right.key = 3
if not has_path(tree, 22):
  print("Correct! The tree no longer has a path of sum 22.")
else:
  print("Incorrect! The tree should no longer have a path of sum 22.")

# Question 17: Find the missing number.

def find_missing(ns):
  l = len(ns)
  if l == 0:
    return -1
  # Using // isn't strictly necessary since this product is always even.
  # We can use l(l+1)/2 because the list is missing an element.
  expected = (l * (l+1)) // 2
  return expected - sum(ns)

missing = find_missing([4, 0, 3, 1])
if missing == 2:
  print("Correct! The missing number is 2.")
else:
  print("Incorrect! The missing number should have been 2, not %d." % missing)

missing = find_missing([8, 3, 5, 2, 4, 6, 0, 1])
if missing == 7:
  print("Correct! The missing number is 7.")
else:
  print("Incorrect! The missing number should have been 7, not %d." % missing)

# Question 18: Reverse a linked list.

def reverse_list(head):
  # This performs an in-place reversal and returns the new head.
  # At each iteration, we unlink the current node and insert it
  # at the beginning of the list.
  new_head = head
  node = head.next
  new_head.next = None
  while node is not None:
    next = node.next
    node.next = new_head
    new_head = node
    node = next
  return new_head

ll = Link(1, Link(2, Link(3, Link(4, Link(5)))))
ll = reverse_list(ll)

ns = [n.key for n in ll]
if ns == [5, 4, 3, 2, 1]:
  print("Correct! The list was reversed.")
else:
  print("Incorrect! The reversed list is not %s." % ns)

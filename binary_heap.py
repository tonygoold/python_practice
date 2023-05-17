class BinaryHeap:
  """A heap implemented as an array-backed binary tree."""
  def __init__(self, is_max = lambda x, y: x >= y):
    """Initialize a heap.
    
    is_max(x, y) should return true if x == max(x, y). The default value will
    create a max-heap.
    """
    self.is_max = is_max
    self.values = []

  def push(self, value):
    """Adds an element to the heap."""
    self.values.append(value)
    index = len(self.values) - 1
    # Bubble up the new value until the heap property is restored.
    while index > 0:
      parent = (index - 1) // 2
      # Stop if heap property satisfied.
      if self.is_max(self.values[parent], self.values[index]):
        break
      (self.values[index], self.values[parent]) = (self.values[parent], self.values[index])
      index = parent

  def pop(self):
    """Remove and return the element at the top of the heap, or None if the
    heap is empty."""
    # Swap the top of the heap with the last element. After removing the top
    # of the heap, end will be the new size of the heap.
    end = len(self.values) - 1
    if end < 0:
      return None
    elif end == 0:
      # Short-circuit on single-item heaps
      return self.values.pop()

    self.values[0], self.values[end] = self.values[end], self.values[0]
    result = self.values.pop()
    # Bubble down the new top of heap until the heap property is restored.
    index = 0
    # Note: The  while condition will never be false
    while index < end:
      left, right = 2 * index + 1, 2 * index + 2
      # Find the index of the max child.
      right_max = right < end and self.is_max(self.values[right], self.values[left])
      max_index = right if right_max else left
      # Stop if heap property satisfied (including if now a leaf node).
      if max_index >= end or self.is_max(self.values[index], self.values[max_index]):
        break
      self.values[index], self.values[max_index] = self.values[max_index], self.values[index]
      index = max_index
    return result
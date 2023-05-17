from linked_list import LinkedList

class LRUCache:
  def __init__(self, capacity):
    self.capacity = capacity
    self.keys = LinkedList()
    self.values = {}
  
  def get(self, key):
    # Combines two O(N) operations into a single O(N) operation.
    if not self.keys.delete_and_append(key):
      return None
    return self.values[key]

  def put(self, key, value):
    # Combines two O(N) operations into a single O(N) operation.
    self.keys.maybe_delete_and_append(key)
    self.values[key] = value
    if len(self.values) > self.capacity:
      del self.values[self.keys.delete_head()]


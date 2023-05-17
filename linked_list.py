class Link:
  def __init__(self, key, next = None):
    self.key = key
    self.next = next

  def __iter__(self):
    node = self
    while node is not None:
      yield node
      node = node.next

class LinkedList:
  def __init__(self):
    self.head = Link(None)
  
  def __iter__(self):
    node = self.head
    while node.next is not None:
      node = node.next
      yield node.key
  
  def size(self):
    count = 0
    node = self.head
    while node.next is not None:
      count += 1
      node = node.next
    return count

  def contains(self, key):
    node = self.head
    while node.next is not None:
      node = node.next
      if node.key == key:
        return True
    return False

  def insert_head(self, key):
    self.head.next = Link(key, self.head.next)

  def append(self, key):
    node = self.head
    while node.next is not None:
      node = node.next
    node.next = Link(key)

  def delete(self, key):
    node = self.head
    while node.next is not None:
      if node.next.key == key:
        node.next = node.next.next
        return True
      node = node.next
    return False

  def delete_head(self):
    if self.head.next is None:
      return None
    node = self.head.next
    self.head.next = node.next
    return node.key
  
  def delete_and_append(self, key):
    """Moves an element to the end of the list. Does nothing if the element is
    not found. This is equivalent to a successful delete and an append, but it
    avoids traversing the list twice."""
    node = self.head
    while node.next is not None:
      if node.next.key == key:
        found = node.next
        node.next = found.next
        found.next = None
        while node.next is not None:
          node = node.next
        node.next = found
        return True
      node = node.next
    return False

  def maybe_delete_and_append(self, key):
    """Moves an element to the end of the list. Appends the element if it is
    not found. This is equivalent to an unconditional delete and an append,
    but it avoids traversing the list twice."""
    node = self.head
    while node.next is not None:
      if node.next.key == key:
        found = node.next
        node.next = found.next
        found.next = None
        while node.next is not None:
          node = node.next
        node.next = found
        return
      node = node.next
    node.next = Link(key)

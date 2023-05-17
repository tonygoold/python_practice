class BinaryNode:
  def __init__(self, key, left = None, right = None):
    self.key = key
    self.left = left
    self.right = right
  
  def preorder(self, fn):
    fn(self)
    if self.left is not None:
      self.left.preorder(fn)
    if self.right is not None:
      self.right.preorder(fn)
  
  def inorder(self, fn):
    if self.left is not None:
      self.left.inorder(fn)
    fn(self)
    if self.right is not None:
      self.right.inorder(fn)

  def postorder(self, fn):
    if self.left is not None:
      self.left.postorder(fn)
    if self.right is not None:
      self.right.postorder(fn)
    fn(self)

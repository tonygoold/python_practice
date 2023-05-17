class Interval:
  def __init__(self, low, high):
    self.low = low
    self.high = high

  def overlaps(self, other: 'Interval') -> bool:
    return (self.low <= other.low and self.high >= other.low) or \
      (other.low <= self.low and other.high >= self.high)

  def merge(self, other: 'Interval') -> 'Interval':
    if self.overlaps(other):
      return Interval(min(self.low, other.low), max(self.high, other.high))
    return None

  def __str__(self):
    return "[%d, %d]" % (self.low, self.high)

  def __lt__(self, other):
    return self.low < other.low or (self.low == other.low and self.high < other.high)

  def __le__(self, other):
    return self.low < other.low or (self.low == other.low and self.high <= other.high)

  def __eq__(self, other):
    return self.low == other.low and self.high == other.high

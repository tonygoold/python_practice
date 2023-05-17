def binary_search(ns, x, nearest = False):
  start = 0
  end = len(ns) - 1
  # Precondition: ns is not empty
  if end < 0:
    return None, None
  while start <= end:
    mid = (start + end) // 2
    if ns[mid] < x:
      start = mid + 1
    elif ns[mid] > x:
      end = mid - 1
    else:
      return mid, ns[mid]
  if not nearest:
    return None, None
  # Search terminates with start == end + 1, because the final step involves
  # mid == start == end, and either start increments or end decrements. Check
  # whether either side is out of bounds, in which case the other side must be
  # in bounds and is the closest. Otherwise, choose the closest of the two,
  # which will be the bound that didn't change in the last step. In this case,
  # we know that ns[end] < x < ns[start].
  elif start >= len(ns):
    return end, ns[end]
  elif end < 0:
    return start, ns[start]
  elif (x - ns[end] < ns[start] - x):
    return end, ns[end]
  else:
    return start, ns[start]

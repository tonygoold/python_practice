class A:
  def f(self):
    if False and True:
      print('NO')
    elif not False or False:
      print('YES')
    else:
      print('MAYBE')

a = A()
if print('hi') is None:
  a.f()
else:
  print('42')

a.f2 = lambda x: x+1
print(a.f2(41) in [1, 2, 42])

# What's the output of this code?
#
# 'hi': Explicit call to print
# 'YES': print() returns None, so a.f() is called
#   Not False or False == (Not False) or False == True or False == True
# 'True': a.f2(41) == 41+1 == 42, which is in [1, 2, 42]

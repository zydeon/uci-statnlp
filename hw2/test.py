class defaultdict(dict):
  def __init__(self, cb):
    self.cb = cb
    super().__init__(self)

  def __getitem__(self, k):
    try:
      return super().__getitem__(k)
    except KeyError:
      return self.cb()




if __name__ == '__main__':
  D = defaultdict(int)
  # D[1] = D[1] + 1
  D[1] += 1
  x = D[2]
  print(D)

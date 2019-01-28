class Variables():
  def __init__(self):
    import traceback
    for line in traceback.format_stack():
        print line.strip()
    self.data = {}
  def __setitem__(self, label, value):
    self.data[label] = value
  def __getitem__(self, label):
    return self.data[label]
  # def __str__(self):
  #   return str(self.data)

def init():
    global variables
    variables = Variables()
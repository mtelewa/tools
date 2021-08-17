import sys

def progressbarstr(x, maxx, len=60):
    xx = x/maxx
    if xx != 1:
      return '|'+int(xx*len)*'#'+((len-int(xx*len))*'-')+'| {:>3}% ' \
             '({}/{})'.format(int(xx*100), x, maxx)
    else:
      return '|'+int(xx*len)*'#'+((len-int(xx*len))*'-')+'| {:>3}% ' \
             '({}/{})\n'.format(int(xx*100), x, maxx)

def progressbar(x, maxx, len=60):
    sys.stdout.write('{}\r'.format(progressbarstr(x, maxx, len=len)))
    sys.stdout.flush()

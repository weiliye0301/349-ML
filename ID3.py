from node import Node
import math

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  best, values = choose_attribute(examples)
  t = Node()
  t.label = mode(examples)
  t.split = best
  if not examples: 
    return default

  elif same_class(examples) or choose_attribute(examples) == None:
    node = Node()
    node.label = mode(examples)
    return node

  else:
    for val in values.keys():
      examples_i = [ex for ex in examples if ex.get(best) == val]
      subtree = ID3(examples_i,mode(examples))
      t.children[val] = subtree

    return t

def choose_attribute(examples):
  key = float('inf')
  for attribute in examples[0].keys():
    if attribute == "Class":
      continue
    exp, v = Entropy(examples,attribute)
    while key > exp:
      attrib_i = attribute
      scores = v
      key = exp      
  return attrib_i, scores

def Entropy(examples, attribute):
  tally = {} 
  tot = {} 
  ent = []
  probs = [] 

  for ex in examples:
    item = ex.get(attribute) 
    cc = ex.get("Class")
    if item in tally:
      tot[item] += 1
      if cc not in tally[item]:
        tally[item][cc] = 1
      else:
        tally[item][cc] += 1
    else:
      tot[item] = 1
      tally[item] = {cc:1}

  for cla in tally:
    for sub in tally[cla]: 
      p_y_x = float(tally[cla][sub])/tot[cla]
      ent.append(-p_y_x*math.log(p_y_x,2))
    probs.append(tot[cla]/len(examples)*sum(ent))

  entropy = sum(probs)
  return entropy, tot

def same_class(examples):
  first_label = examples[0].get("Class")

  for ex in examples:
    while ex.get("Class") != first_label:
      return False
  return True

def mode(examples):
  mode_Dic = {}
  for ex in examples:
    if ex['Class'] not in mode_Dic.keys():
      mode_Dic[ex['Class']] = 0
    else:
      mode_Dic[ex['Class']] += 1
  return max(mode_Dic, key=mode_Dic.get)

def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''
  

def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  valid = 0.0
  for example in examples:
    ex_class = evaluate(node, example)
    ex_true = example['Class']
    if ex_class == ex_true:
      valid += 1.0
  num_valid = valid/ len(examples)
  return num_valid


def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''
  if node.children=={} or example[node.split] not in node.children:
      return node.label
  
  attrib = node.split
  key = example[attrib]
  return evaluate(node.children[key],example)


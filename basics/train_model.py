from nn import MLP

print('Initiating the MLP')
model = MLP(3,[5,5,1])

print('Run the model on sample inputs to get the output')
x = [1,3,0.5]
print(model(x))

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
  [2.0, 4.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0, 1.5]

# train the model for 20 epochs
for k in range(200):
  ypred = [model(x) for x in xs]
  loss = sum((yPred - yActual)**2 for yActual, yPred in zip(ys, ypred))
  # backward pass
  for p in model.parameters():
    p.grad = 0.0
  # calculate the gradients (how each neuron affects the final y value)
  loss.backward()
  # update
  for p in model.parameters():
    p.data += -0.01 * p.grad
  print(k, loss.data)

# y predictions are close to expected.
print(ypred)

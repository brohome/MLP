import mlp as m
import pandas as pd

train = pd.read_csv('./example/data/train2.csv')
test = pd.read_csv('./example/data/test2.csv')
x = train.drop(columns='7')
y = train['7']

data = m.utils.DataLoader(x, y, batch_size=256)

layers = [
    m.Dense(7, 64),
    m.ReLu(),
    m.Dense(64, 1),
]

model = m.MLP(layers, learning_rate=0.02)
loss_func = m.mse_loss
optim = m.SGD(model.parameters(), learning_rate=0.05)
n_epochs = 5

for epoch in range(n_epochs):
    for x_batch, y_batch in data.iterate_batches():
        out = model.forward(x_batch) # Получаем предикт
        loss, grad = loss_func(out, y_batch) # Считаем ошибку
        model.backward(grad) # Считаем градиент и корректируем веса
        optim.step()
    print("MSE: ", loss)
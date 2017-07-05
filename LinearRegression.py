import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#The numbers of data
num_points = 1000

# DataSet
vectors_set = []

for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = 0.1 * x1 + 0.3 + np.random.normal(0.0, 0.03) # h(x) = theta * x + b
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

theta = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = theta * x_data + b

# 제곱한 값의 평균을 구함
loss = tf.reduce_mean(tf.square(y - y_data))

# Gradient Descent 적용
# Learning Rate = 0.5
optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)

# 알고리즘 실행
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


plt.show()
for epoch in range(100):
    sess.run(train)
    print (epoch ,sess.run(loss), sess.run(theta), sess.run(b)) # step, J(cost), theta, b

plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-2, 2)
plt.ylim(0.1, 0.6)
plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(theta) * x_data + sess.run(b))
plt.show()
print (sess.run(theta), sess.run(b))





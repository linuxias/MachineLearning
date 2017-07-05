import tensorflow as tf

#Data
x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]

# PlaceHolder
# shape : None - variable, 3 - data count
# 즉 x_data를 보면, List 내에 들어갈 수 있는 List의 개수는 정해지지 않고(None),
# List 내의 data는 3개인 x란 이름의 PlaceHolder 생성
x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

theta = tf.Variable(tf.random_normal([3, 1]), name='theta')
bias = tf.Variable(tf.random_normal([1]), name='bias')

#Hypothesis Model
hypothesis = tf.matmul(x, theta) + bias

#Cost function
cost = tf.reduce_mean(tf.square(hypothesis - y))

#GradientDescent
learning_rate = 1e-5
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# epoch
# feed_dict - PlaceHolder에 data update를 위함.
for epoch in range(10001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                          feed_dict={x: x_data, y: y_data})

    if epoch % 10 == 0:
        print(epoch, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

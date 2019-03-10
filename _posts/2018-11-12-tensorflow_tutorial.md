tensorflow 감을 잡기 위한 튜토리얼  
코드와 사진을 동시에 해놓는 게 이해하기 쉬운 듯하다.  

# 데이터 준비

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```
필기체로 된 숫자들의 데이터(MNIST)를 불러온다.
![](https://i.imgur.com/y0cHe8V.png)


```python
mnist
```




    Datasets(train=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x0000015D0418B710>, validation=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x0000015D0C710F28>, test=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x0000015D0C710F60>)



# Set parameters
하이퍼파라미터를 지정


```python
import tensorflow as tf
learning_rate = 0.01
training_iteration = 3
batch_size = 100
display_step = 2
```

learning_rate
![](https://i.imgur.com/ROD6Fnb.png)

# TF graph input
예를 들어 label이 1인 sample 하나의 생김새는 이렇다
![](https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/images/MNIST-Matrix.png)

28*28이라 일자로 펼치면 784개가 된다.  
![](https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/images/mnist-train-xs.png)

tensorflow에서 연산할 때 입력받는 공간을 placeholder라 한다.  
784개의 벡터를 2차원 텐서를 넣을 예정이다.  
None은 길이를 임의지정이 가능하게 정의하는 것  


```python
x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes
```

![](https://i.imgur.com/ag5VN2C.png)

# Create a model


```python
# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
```

![](https://i.imgur.com/KDQ6eCN.png)


```python
with tf.name_scope("Wx_b") as scope:
    # Construct a linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
```

![](https://i.imgur.com/IZLm5PO.png)


```python
# Add summary ops to collect data
w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b)
```

나중에 텐서보드를 이용해서 쉽게 확인할 수 있다.
![](https://i.imgur.com/Su8pkJW.png)


```python
# More name scopes will clean up graph representation
with tf.name_scope("cost_function") as scope:
    # Minimize error using cross entropy
    # Cross entropy
    cost_function = -tf.reduce_sum(y*tf.log(model))
    # Create a summary to monitor the cost function
    tf.summary.scalar("cost_function", cost_function)

with tf.name_scope("train") as scope:
    # Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
```

![](https://i.imgur.com/GMIOwAX.png)


```python
# Initializing the variables
init = tf.global_variables_initializer()

# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()

# Launch the graph
tensorflow는 graph를 통해 모델이 실행되는 듯 하다.  
```

```python
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute the average loss
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch

        # Display logs per iteration step
        if iteration % display_step == 0:
            print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Tuning completed!")

    # Test the model
    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
```

    Iteration: 0001 cost= 30.536476088
    Iteration: 0003 cost= 21.075364939
    Tuning completed!
    Accuracy: 0.9168
    
